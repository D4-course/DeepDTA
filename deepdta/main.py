import argparse
import sys


import deepdta.DTI as models
from deepdta.dataset import load_process_DAVIS
from deepdta.utils import (
    data_process,
    generate_config,
    is_valid_smiles,
    is_valid_protein,
)

drug_encoding = "CNN"
target_encoding = "CNN"


def train_model():
    X_drug, X_target, y = load_process_DAVIS("./data/", binary=False)
    train, val, test = data_process(
        X_drug,
        X_target,
        y,
        drug_encoding,
        target_encoding,
        split_method="random",
        frac=[0.7, 0.1, 0.2],
    )

    # use the parameters setting provided in the paper: https://arxiv.org/abs/1801.10193
    config = generate_config(
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
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


def predict_dta(X_drug, X_target):
    y = [0] * (len(X_drug) * len(X_target))
    X_pred = data_process(
        X_drug, X_target, y, drug_encoding, target_encoding, split_method="no_split"
    )
    model = models.model_pretrained("./weights/")
    y_pred = model.predict(X_pred)
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
    parser = arg_parser()
    X_drug = parser.parse_args().drug
    X_target = parser.parse_args().target
    if not X_drug:
        print("Please provide a drug SMILES")
        sys.exit(0)
    if not X_target:
        print("Please provide a target(protein) sequence")
        sys.exit(0)
    if not is_valid_smiles(X_drug):
        print(f"Invalid drug SMILES - {X_drug}")
        sys.exit(0)
    if not is_valid_protein(X_target):
        print(f"Invalid protein sequence - {X_target}")
        sys.exit(0)
    try:
        y_pred = predict_dta([X_drug], [X_target])
        print(f"Predicted affinity: {y_pred[0]}")
    except Exception as err:
        print(err)
        sys.exit(0)


if __name__ == "__main__":
    main()
