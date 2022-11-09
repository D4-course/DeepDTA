# from DeepPurpose import models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from DeepPurpose import DTI as models
import argparse

def arg_parser():
    """Read arguments from command line"""

    parser = argparse.ArgumentParser(
        description="""DeepDTA: Deep Learning for Drug Target Interaction Prediction""")
    
    parser.add_argument("--drug", type=str, help="drug SMILES", required=True)
    parser.add_argument(
        "--target", type=str, help="Target Protein", default="O"
    )
    return parser

def main():
    """Entypoint function"""
    parser = arg_parser()
    drug = parser.parse_args().drug
    target = parser.parse_args().target

    X_drug, X_target, y = load_process_DAVIS('./data/', binary=False)

    ### Types of Drug and Target Representation available
    # 'MPNN', 'CNN', 'Transformer', 'RNN', 'AttentiveFP', 'Weave', 'Morgan', 'RDKit2D', 'ECFP', 'GAT', 'GCN', 'GraphConv', 'DGCNN', 'GIN', 'SAGE'

    drug_encoding = 'CNN'
    target_encoding = 'CNN'

    net = models.model_pretrained(model='CNN_CNN_DAVIS')

    X_drug = [drug]
    X_target = [target]
    y = [7.365]
    X_pred = data_process(X_drug, X_target, y, 
                                    drug_encoding, target_encoding, 
                                    split_method='no_split')
    y_pred = net.predict(X_pred)
    print('The predicted score is ' + str(y_pred))

if __name__ == "__main__":
    main()