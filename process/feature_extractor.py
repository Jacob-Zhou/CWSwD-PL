from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class FeatureExtractor(object):
    def extract(self, tokens):
        raise NotImplementedError()

    def restore(self, ids):
        raise NotImplementedError()
