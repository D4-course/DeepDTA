""" Main module to predict binding affinity or to train the model"""

import argparse
import sys


import deepdta.DTI as models
from deepdta.dataset import load_process_davis
from deepdta.utils import (
    data_process,
    generate_config,
    is_valid_smiles,
    is_valid_protein,
)

DRUG_ENCODING = "CNN"
TARGET_ENCODING = "CNN"


def train_model():
    """Train the model"""
    var_x_drug, var_x_target, var_y = load_process_davis("./data/")
    train, val, test = data_process(
        var_x_drug,
        var_x_target,
        var_y,
        DRUG_ENCODING,
        TARGET_ENCODING,
        split_method="random",
        frac=[0.7, 0.1, 0.2],
    )

    # use the parameters setting provided in the paper: https://arxiv.org/abs/1801.10193
    config = generate_config(
        drug_encoding=DRUG_ENCODING,
        target_encoding=TARGET_ENCODING,
        cls_hidden_dims=[1024, 1024, 512],
        train_epoch=100,
        LR=0.001,
        batch_size=256,
        cnn_drug_filters=[32, 64, 96],
        cnn_target_filters=[32, 64, 96],
        cnn_drug_kernels=[4, 6, 8],
        cnn_target_kernels=[4, 8, 12],
    )

    model = models.model_initialize(**config)
    model.train(train, val, test)
    model.save_model("./weights/")


def predict_dta(var_x_drug, var_x_target):
    """Predict binding affinity"""
    var_y = [0] * (len(var_x_drug) * len(var_x_target))
    var_x_pred = data_process(
        var_x_drug,
        var_x_target,
        var_y,
        DRUG_ENCODING,
        TARGET_ENCODING,
        split_method="no_split",
    )
    model = models.model_pretrained("./weights/")
    y_pred = model.predict(var_x_pred)
    return y_pred


def arg_parser():
    """Read arguments from command line"""

    parser = argparse.ArgumentParser(
        description="""DeepDTA: Deep Drug-Target Binding Affinity Prediction"""
    )
    parser.add_argument("--drug", type=str, help="Drug SMILES", required=True)
    parser.add_argument(
        "--target", type=str, help="Target(protein) sequence", required=True
    )
    return parser


def main():
    """Entrypoint function"""
    parser = arg_parser()
    var_x_drug = parser.parse_args().drug
    var_x_target = parser.parse_args().target
    if not var_x_drug:
        print("Please provide a drug SMILES")
        sys.exit(0)
    if not var_x_target:
        print("Please provide a target(protein) sequence")
        sys.exit(0)
    if not is_valid_smiles(var_x_drug):
        print(f"Invalid drug SMILES - {var_x_drug}")
        sys.exit(0)
    if not is_valid_protein(var_x_target):
        print(f"Invalid protein sequence - {var_x_target}")
        sys.exit(0)
    try:
        y_pred = predict_dta([var_x_drug], [var_x_target])
        print(f"Predicted affinity: {y_pred[0]}")
    except Exception as err:
        print(err)
        sys.exit(0)


if __name__ == "__main__":
    main()
