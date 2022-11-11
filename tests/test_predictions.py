import pytest
import deepdta.DTI as models
from deepdta.utils import data_process
from tests.inputs import sample_drug, sample_target, output, drug_encoding, target_encoding


def test_one():
    "Load Model"
    model = models.model_pretrained("./weights/")
    y = [0] * (len(sample_drug) * len(sample_target))
    X_pred = data_process(
        sample_drug, sample_target, y, drug_encoding, target_encoding, split_method="no_split"
    )
    y_pred = model.predict(X_pred)
    assert y_pred[0] == output
