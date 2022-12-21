from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
import logging
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
from sentence_transformers.readers import InputExample
import torch


logger = logging.getLogger(__name__)


def frob(A, B):
    return torch.sqrt(torch.diagonal(A @ B.permute(0, 2, 1) @ B @ A.permute(0, 2, 1), dim1=-2, dim2=-1).sum(-1))


def frob_cos(A, B):
    return frob(A, B) / (frob(A, A).sqrt() * frob(B, B).sqrt())


class MatrixEmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)

        labels = self.scores

        schatten_distances = []
        for i in range(len(embeddings1)):
            a = torch.tensor(embeddings1[i]).unsqueeze(0).permute(0, 2, 1)
            b = torch.tensor(embeddings2[i]).unsqueeze(0).permute(0, 2, 1)

            sim = frob_cos(a, b).item()
            schatten_distances.append(sim)

        schatten_distances = np.array(schatten_distances)

        eval_pearson_schatten, _ = pearsonr(labels, schatten_distances)
        eval_spearman_schatten, _ = spearmanr(labels, schatten_distances)

        print(eval_spearman_schatten)

        logging.info("Schatten-Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_schatten, eval_spearman_schatten))

        return eval_spearman_schatten
