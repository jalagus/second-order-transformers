import torch
import torch.nn as nn


class SVDLayer(nn.Module):
    def __init__(self, k=0):
        super(SVDLayer, self).__init__()
        self.k = k

    def forward(self, features):
        token_embeddings = features['token_embeddings']
        out = token_embeddings

        if self.k > 0:
            _, s, v = torch.svd_lowrank(out, q=self.k + 2)
            features.update({'sentence_embedding': (torch.diag_embed(s) @ v.permute([0, 2, 1])).permute([0, 2, 1])[:, :, :self.k]})
        else:
            features.update({'sentence_embedding': out})
        return features

    def save(self, output_path):
        pass

    @staticmethod
    def load(input_path: str):
        return SVDLayer()
