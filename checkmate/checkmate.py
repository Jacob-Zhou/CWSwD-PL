import os
import glob
import json
import numpy as np
import tensorflow as tf


class BestCheckpointSaver(object):
    """Maintains a directory containing only the best n checkpoints

    Inside the directory is a best_checkpoints JSON file containing a dictionary
    mapping of the best checkpoint filepaths to the values by which the checkpoints
    are compared.  Only the best n checkpoints are contained in the directory and JSON file.

    This is a light-weight wrapper class only intended to work in simple,
    non-distributed settings.  It is not intended to work with the tf.Estimator
    framework.
    """

    def __init__(self, save_dir, num_to_keep=1, maximize=True, saver=None):
        """Creates a `BestCheckpointSaver`

        `BestCheckpointSaver` acts as a wrapper class around a `tf.train.Saver`

        Args:
            save_dir: The directory in which the checkpoint files will be saved
            num_to_keep: The number of best checkpoint files to retain
            maximize: Define 'best' values to be the highest values.  For example,
              set this to True if selecting for the checkpoints with the highest
              given accuracy.  Or set to False to select for checkpoints with the
              lowest given error rate.
            saver: A `tf.train.Saver` to use for saving checkpoints.  A default
              `tf.train.Saver` will be created if none is provided.
        """
        self._num_to_keep = num_to_keep
        self._save_dir = save_dir
        self._maximize = maximize
        self._saver = saver if saver else tf.train.Saver(
            max_to_keep=None,
            save_relative_paths=True
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.best_checkpoints_file = os.path.join(save_dir, 'checkpoints_info.json')
        if os.path.exists(self.best_checkpoints_file):
            os.remove(self.best_checkpoints_file)

    def handle(self, value, sess, epoch, prefix=None):
        """Updates the set of best checkpoints based on the given result.

        Args:
            value: The value by which to rank the checkpoint.
            sess: A tf.Session to use to save the checkpoint
            epoch: A `tf.Tensor` represent the global step
            prefix:
        """
        if prefix:
            current_ckpt = os.path.join(self._save_dir, '{}_model.ckpt-{}'.format(prefix, epoch))
            save_path = os.path.join(self._save_dir, '{}_model.ckpt'.format(prefix))

        else:
            current_ckpt = os.path.join(self._save_dir, 'model.ckpt-{}'.format(epoch))
            save_path = os.path.join(self._save_dir, 'model.ckpt')
        value = float(value)
        if not os.path.exists(self.best_checkpoints_file):
            self._save_best_checkpoints_file({current_ckpt: value})
            self._saver.save(sess, save_path, global_step=epoch)
            return True

        best_checkpoints = self._load_best_checkpoints_file()

        if len(best_checkpoints) < self._num_to_keep:
            best_checkpoints[current_ckpt] = value
            self._save_best_checkpoints_file(best_checkpoints)
            self._saver.save(sess, save_path, global_step=epoch)
            return True

        if self._maximize:
            should_save = not all(current_best >= value
                                  for current_best in best_checkpoints.values())
        else:
            should_save = not all(current_best <= value
                                  for current_best in best_checkpoints.values())
        if should_save:
            best_checkpoint_list = self._sort(best_checkpoints)

            worst_checkpoint = best_checkpoint_list.pop(-1)[0]
            self._remove_outdated_checkpoint_files(worst_checkpoint)
            self._update_internal_saver_state(best_checkpoint_list)

            best_checkpoints = dict(best_checkpoint_list)
            best_checkpoints[current_ckpt] = value
            self._save_best_checkpoints_file(best_checkpoints)
            self._saver.save(sess, save_path, global_step=epoch)

        return should_save

    def _save_best_checkpoints_file(self, updated_best_checkpoints):
        with open(self.best_checkpoints_file, 'w') as f:
            json.dump(updated_best_checkpoints, f, indent=3)

    def _remove_outdated_checkpoint_files(self, worst_checkpoint):
        os.remove(os.path.join(self._save_dir, 'checkpoint'))
        for ckpt_file in glob.glob(worst_checkpoint + '.*'):
            os.remove(ckpt_file)

    def _update_internal_saver_state(self, best_checkpoint_list):
        best_checkpoint_files = [
            (ckpt[0], np.inf)  # TODO: Try to use actual file timestamp
            for ckpt in best_checkpoint_list
        ]
        self._saver.set_last_checkpoints_with_time(best_checkpoint_files)

    def _load_best_checkpoints_file(self):
        with open(self.best_checkpoints_file, 'r') as f:
            best_checkpoints = json.load(f)
        return best_checkpoints

    def _sort(self, best_checkpoints):
        best_checkpoints = [
            (ckpt, best_checkpoints[ckpt])
            for ckpt in sorted(best_checkpoints,
                               key=best_checkpoints.get,
                               reverse=self._maximize)
        ]
        return best_checkpoints


def get_best_checkpoint(best_checkpoint_dir, select_maximum_value=True):
    """ Returns filepath to the best checkpoint

    Reads the best_checkpoints file in the best_checkpoint_dir directory.
    Returns the filepath in the best_checkpoints file associated with
    the highest value if select_maximum_value is True, or the filepath
    associated with the lowest value if select_maximum_value is False.

    Args:
        best_checkpoint_dir: Directory containing best_checkpoints JSON file
        select_maximum_value: If True, select the filepath associated
          with the highest value.  Otherwise, select the filepath associated
          with the lowest value.

    Returns:
        The full path to the best checkpoint file

    """
    best_checkpoints_file = os.path.join(best_checkpoint_dir, 'checkpoints_info.json')
    checkpoint_file = os.path.join(best_checkpoint_dir, 'checkpoint')
    if os.path.exists(best_checkpoints_file):
        with open(best_checkpoints_file, 'r') as f:
            best_checkpoints = json.load(f)
        best_checkpoints = [
            (int(ckpt.split("-")[-1]), ckpt) for ckpt in sorted(best_checkpoints)
        ]
        return best_checkpoints[0]
    else:
        assert os.path.exists(checkpoint_file)
        with open(checkpoint_file, 'r') as f:
            first_line = f.readline()
            line_data = first_line.split()
            assert line_data[0] == "model_checkpoint_path:"
            return 0, os.path.join(best_checkpoint_dir, line_data[1][1:-1])


def get_all_checkpoint(best_checkpoint_dir):
    best_checkpoints_file = os.path.join(best_checkpoint_dir, 'checkpoints_info.json')
    checkpoint_file = os.path.join(best_checkpoint_dir, 'checkpoint')
    if os.path.exists(best_checkpoints_file):
        with open(best_checkpoints_file, 'r') as f:
            best_checkpoints = json.load(f)
        best_checkpoints = [
            (int(ckpt.split("-")[-1]), ckpt) for ckpt in sorted(best_checkpoints)
        ]
        return best_checkpoints
    else:
        assert os.path.exists(checkpoint_file)
        with open(checkpoint_file, 'r') as f:
            first_line = f.readline()
            line_data = first_line.split()
            assert line_data[0] == "model_checkpoint_path:"
            return [(0, os.path.join(best_checkpoint_dir, line_data[1][1:-1]))]
