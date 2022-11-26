from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from deepdta.utils import is_valid_smiles, is_valid_protein
from deepdta.main import predict_dta

# import model

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def main(drug, target):
    var_x_drug = drug
    var_x_target = target
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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/prediction")
def prediction(drug, target):
    var_x_drug = drug
    var_x_target = target
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
    except Exception as err:
        print(err)
    return {"Predicted affinity": y_pred[0]}