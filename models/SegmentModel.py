class SegmentModel(object):

    def __init__(self):
        self.is_training = True
        self.seq_length = 0
        self.prediction = None
        self.label_ids = None
        self.loss = None
        self.log_likelihood = None

    def get_all_results(self):
        return self.loss, self.label_ids, self.prediction, self.seq_length