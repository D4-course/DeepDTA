"""Helper module for loading and processing Kinnase DAVID dataset."""

import json
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
import wget

from deepdta.utils import convert_y_unit


def load_process_davis(path="./data", convert_to_log=True):
    """Load and process Kinnase DAVIS dataset."""
    print("Beginning Processing...")

    if not os.path.exists(path):
        os.makedirs(path)

    url = "https://github.com/futianfan/DeepPurpose_Data/blob/main/DAVIS.zip?raw=true"
    saved_path = wget.download(url, path)

    print("Beginning to extract zip file...")
    with ZipFile(saved_path, "r") as zipfile:
        zipfile.extractall(path=path)

    affinity = pd.read_csv(path + "/DAVIS/affinity.txt", header=None, sep=" ")

    with open(path + "/DAVIS/target_seq.txt", encoding="utf8") as file:
        target = json.load(file)

    with open(path + "/DAVIS/SMILES.txt", encoding="utf8") as file:
        drug = json.load(file)

    target = list(target.values())
    drug = list(drug.values())

    smiles = []
    target_seq = []
    var_y = []

    for i, smile in enumerate(drug):
        for j, seq in enumerate(target):
            smiles.append(smile)
            target_seq.append(seq)
            var_y.append(affinity.values[i, j])

    if convert_to_log:
        print("Default set to logspace (nM -> p) for easier regression")
        var_y = convert_y_unit(np.array(var_y), "nM", "p")
    else:
        pass
    print("Done!")
    return np.array(smiles), np.array(target_seq), np.array(var_y)
