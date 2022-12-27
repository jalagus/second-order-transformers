import numpy as np
import torch
from pooling_lib.SVDLayer import SVDLayer
from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers.readers import InputExample
import os
import gzip
import csv
from pooling_lib.MatrixEmbeddingSimilarityEvaluator import MatrixEmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import math
from pooling_lib.SchattenSimilarityLoss import SchattenSimilarityLoss

"""
Modified from the SBERT example in
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark.py
"""

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    sts_dataset_path = "data/stsbenchmark.tsv.gz"

    if not os.path.exists(sts_dataset_path):
        util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)

    word_embedding_model = models.Transformer("bert-base-uncased")
    svd = SVDLayer(k=5)
    model = SentenceTransformer(modules=[word_embedding_model, svd])

    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0
            inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

            if row["split"] == "dev":
                dev_samples.append(inp_example)
            elif row["split"] == "test":
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
    train_loss = SchattenSimilarityLoss(model=model)

    evaluator = MatrixEmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="sts-dev")

    num_epochs = 10

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
    )

    test_evaluator = MatrixEmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")
    test_evaluator(model)
