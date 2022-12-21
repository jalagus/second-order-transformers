import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer


def schatten(A, B, p: float = 2.0):
    A = A.cpu()
    B = B.cpu()

    if p == 1.0:
        _, s, _ = torch.svd(A @ B.permute(0, 2, 1))
        return s.sum(1)
    elif p > 1:
        _, s, _ = torch.svd(A @ B.permute(0, 2, 1))
        return (s ** p).sum(1) ** (1 / p)
    else:
        _, s, _ = torch.svd(A @ B.permute(0, 2, 1))
        return s.max(1)


class SchattenSimilarityLoss(nn.Module):
    def __init__(self, model: SentenceTransformer):
        super(SchattenSimilarityLoss, self).__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['token_embeddings'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        output = schatten(rep_a, rep_b) / (schatten(rep_a, rep_a).sqrt() * schatten(rep_b, rep_b).sqrt())
        output = output
        loss_fct = nn.MSELoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output
