# -*- coding: utf-8 -*-
import tensorflow as tf

from models.model_utils import CRF
from .SegmentModel import SegmentModel
import torch
from torch import nn
from torch import optim
from torch.nn import functional
from models import model_utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class PytorchModel(nn.Module):
    '''
    Baseline models
    BiLSTM+CRF and Stacked BiLSTM+CRF
    '''
    def __init__(self, config, features, dropout_keep_prob, init_embedding=None, bi_embedding=None):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. rue for training model, false for eval model. Controls
            whether dropout will be applied.
          input_ids: int64 Tensor of shape [batch_size, seq_length, feat_size].
          label_ids: (optional) int64 Tensor of shape [batch_size, seq_length].
          seq_length: (optional) int64 Tensor of shape [batch_size].
          init_embedding: (optional)

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """

        super(PytorchModel, self).__init__()
        dropout_prob = config.hidden_dropout_prob

        if init_embedding is None:
            self.embedding = nn.Embedding(config.vocab_size,
                                          config.embedding_size,
                                          padding_idx=0
                                          )
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(init_embedding), freeze=False)

        self.dropout = nn.Dropout(dropout_prob)

        self.lstm = nn.LSTM(
            config.embedding_size,
            config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob
        )

        self.scores = nn.Linear(config.hidden_size * 2, config.num_classes)

        self.crf = CRF(config.num_classes)

    def forward(self, features):
        input_ids = features["input_ids"]
        seq_length = features["seq_length"]
        # input_ids : (B, L)
        input_ids = torch.tensor(input_ids).long()
        seq_length = torch.tensor(seq_length).long()

        x = self.embedding(input_ids)  # (B, L, C)
        x = self.dropout(x)

        pack_embed = pack_padded_sequence(x, seq_length, True)
        pack_feat, (hidden, _) = self.lstm(pack_embed)
        feat, _ = pad_packed_sequence(pack_feat, True)
        feat = self.dropout(feat)
        scores = self.scores(feat)
        return scores  #, hidden

    def NLLLoss(self, features):
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]
        seq_length = torch.tensor(seq_length).long()
        label_ids = torch.tensor(label_ids).long()
        scores = self(features)                # (B, L, C)
        scores = scores.transpose(0, 1)                 # (L, B, C)
        batch_label, mask = label_ids.t()   # (B, L) -> (L, B)
        loss = self.crf(scores, batch_label, seq_length)
        return loss

    @torch.no_grad()
    def predict(self, features):
        seq_length = features["seq_length"]
        seq_length = torch.tensor(seq_length).long()
        self.eval()
        scores = self(features)          # (B, L, C)
        scores = scores.transpose(0, 1)                 # (L, B, C)
        predict = self.crf.viterbi(scores, seq_length)
        return predict

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)

