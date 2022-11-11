import json
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
import wget

from deepdta.utils import convert_y_unit


def load_process_DAVIS(path="./data", binary=False, convert_to_log=True, threshold=30):
    print("Beginning Processing...")

    if not os.path.exists(path):
        os.makedirs(path)

    url = "https://github.com/futianfan/DeepPurpose_Data/blob/main/DAVIS.zip?raw=true"
    saved_path = wget.download(url, path)

    print("Beginning to extract zip file...")
    with ZipFile(saved_path, "r") as zipfile:
        zipfile.extractall(path=path)

    affinity = pd.read_csv(path + "/DAVIS/affinity.txt", header=None, sep=" ")

    with open(path + "/DAVIS/target_seq.txt") as f:
        target = json.load(f)

    with open(path + "/DAVIS/SMILES.txt") as f:
        drug = json.load(f)

    target = list(target.values())
    drug = list(drug.values())

    SMILES = []
    Target_seq = []
    y = []

    for i in range(len(drug)):
        for j in range(len(target)):
            SMILES.append(drug[i])
            Target_seq.append(target[j])
            y.append(affinity.values[i, j])

    if binary:
        print(
            'Default binary threshold for the binding affinity scores are 30, you can adjust it by using the "threshold" parameter'
        )
        y = [1 if i else 0 for i in np.array(y) < threshold]
    else:
        if convert_to_log:
            print("Default set to logspace (nM -> p) for easier regression")
            y = convert_y_unit(np.array(y), "nM", "p")
        else:
            pass
    print("Done!")
    return np.array(SMILES), np.array(Target_seq), np.array(y)
