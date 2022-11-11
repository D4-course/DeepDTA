"""Module containing the DeepDTA model."""


import copy
import os
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index
from prettytable import PrettyTable
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from deepdta.encoders import CNN
from deepdta.utils import data_process_loader, load_dict, save_dict

torch.manual_seed(2)
np.random.seed(3)


class Classifier(nn.Sequential):
    def __init__(self, model_drug, model_protein, **config):
        super().__init__()
        self.input_dim_drug = config["hidden_dim_drug"]
        self.input_dim_protein = config["hidden_dim_protein"]

        self.model_drug = model_drug
        self.model_protein = model_protein

        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = config["cls_hidden_dims"]
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]

        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)]
        )

    def forward(self, v_D, v_P):
        # each encoding
        v_D = self.model_drug(v_D)
        v_P = self.model_protein(v_P)
        # concatenate and classify
        v_f = torch.cat((v_D, v_P), 1)
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f


def model_initialize(**config):
    model = DBTA(**config)
    return model


def model_pretrained(path_dir, model=None):
    config = load_dict(path_dir)
    model = DBTA(**config)
    model.load_pretrained(path_dir + "/model.pt")
    return model


class DBTA:
    """
    Drug Target Binding Affinity
    """

    def __init__(self, **config):
        drug_encoding = config["drug_encoding"]
        target_encoding = config["target_encoding"]

        if drug_encoding == "CNN":
            self.model_drug = CNN("drug", **config)
        else:
            raise AttributeError("Please use one of the available encoding method.")

        if target_encoding == "CNN":
            self.model_protein = CNN("protein", **config)
        else:
            raise AttributeError("Please use one of the available encoding method.")

        self.model = Classifier(self.model_drug, self.model_protein, **config)
        self.config = config

        if "cuda_id" in self.config:
            if self.config["cuda_id"] is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = torch.device(
                    "cuda:" + str(self.config["cuda_id"])
                    if torch.cuda.is_available()
                    else "cpu"
                )
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.drug_encoding = drug_encoding
        self.target_encoding = target_encoding
        self.result_folder = config["result_folder"]
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        self.binary = False
        if "num_workers" not in self.config.keys():
            self.config["num_workers"] = 0
        if "decay" not in self.config.keys():
            self.config["decay"] = 0

    def test_(self, data_generator, model, repurposing_mode=False):
        y_pred = []
        y_label = []
        model.eval()
        for _, (v_d, v_p, label) in enumerate(data_generator):
            v_d = v_d.float().to(self.device)
            v_p = v_p.float().to(self.device)
            score = self.model(v_d, v_p)
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)
            loss = loss_fct(
                n,
                Variable(torch.from_numpy(np.array(label)).float()).to(self.device),
            )
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to("cpu").numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
        model.train()
        if repurposing_mode:
            return y_pred
        return (
            mean_squared_error(y_label, y_pred),
            pearsonr(y_label, y_pred)[0],
            pearsonr(y_label, y_pred)[1],
            concordance_index(y_label, y_pred),
            y_pred,
            loss,
        )

    def train(self, train, val=None, test=None, verbose=True):

        lr = self.config["LR"]
        decay = self.config["decay"]
        BATCH_SIZE = self.config["batch_size"]
        train_epoch = self.config["train_epoch"]
        loss_history = []

        self.model = self.model.to(self.device)

        # support multiple GPUs
        if torch.cuda.device_count() > 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
            self.model = nn.DataParallel(self.model, dim=0)
        elif torch.cuda.device_count() == 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
        else:
            if verbose:
                print("Let's use CPU/s!")
        # Future TODO: support multiple optimizers with parameters
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        if verbose:
            print("--- Data Preparation ---")

        params = {
            "batch_size": BATCH_SIZE,
            "shuffle": True,
            "num_workers": self.config["num_workers"],
            "drop_last": False,
        }

        training_generator = data.DataLoader(
            data_process_loader(
                train.index.values, train.Label.values, train, **self.config
            ),
            **params,
        )
        if val is not None:
            validation_generator = data.DataLoader(
                data_process_loader(
                    val.index.values, val.Label.values, val, **self.config
                ),
                **params,
            )

        if test is not None:
            info = data_process_loader(
                test.index.values, test.Label.values, test, **self.config
            )
            params_test = {
                "batch_size": BATCH_SIZE,
                "shuffle": False,
                "num_workers": self.config["num_workers"],
                "drop_last": False,
                "sampler": SequentialSampler(info),
            }

            testing_generator = data.DataLoader(
                data_process_loader(
                    test.index.values, test.Label.values, test, **self.config
                ),
                **params_test,
            )

        # early stopping
        max_mse = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ["# epoch"]
        valid_metric_header.extend(
            ["MSE", "Pearson Correlation", "with p-value", "Concordance Index"]
        )
        table = PrettyTable(valid_metric_header)
        float2str = lambda x: "%0.4f" % x
        if verbose:
            print("--- Go for Training ---")
        writer = SummaryWriter()
        t_start = time()
        iteration_loss = 0
        for epo in range(train_epoch):
            for i, (v_d, v_p, label) in enumerate(training_generator):
                v_p = v_p.float().to(self.device)
                v_d = v_d.float().to(self.device)
                # score = self.model(v_d, v_p.float().to(self.device))
                score = self.model(v_d, v_p)
                label = Variable(torch.from_numpy(np.array(label)).float()).to(
                    self.device
                )

                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1)
                loss = loss_fct(n, label)
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()

                if verbose:
                    if i % 100 == 0:
                        t_now = time()
                        print(
                            "Training at Epoch "
                            + str(epo + 1)
                            + " iteration "
                            + str(i)
                            + " with loss "
                            + str(loss.cpu().detach().numpy())[:7]
                            + ". Total time "
                            + str(int(t_now - t_start) / 3600)[:7]
                            + " hours"
                        )
                        ### record total run time

            if val is not None:
                ##### validate, select the best model up to now
                with torch.set_grad_enabled(False):
                    ### regression: MSE, Pearson Correlation, with p-value, Concordance Index
                    mse, r2, p_val, CI, logits, loss_val = self.test_(
                        validation_generator, self.model
                    )
                    lst = ["epoch " + str(epo)] + list(
                        map(float2str, [mse, r2, p_val, CI])
                    )
                    valid_metric_record.append(lst)
                    if mse < max_mse:
                        model_max = copy.deepcopy(self.model)
                        max_mse = mse
                    if verbose:
                        print(
                            "Validation at Epoch "
                            + str(epo + 1)
                            + " with loss:"
                            + str(loss_val.item())[:7]
                            + ", MSE: "
                            + str(mse)[:7]
                            + " , Pearson Correlation: "
                            + str(r2)[:7]
                            + " with p-value: "
                            + str(f"{p_val:.2E}")
                            + " , Concordance Index: "
                            + str(CI)[:7]
                        )
                        writer.add_scalar("valid/mse", mse, epo)
                        writer.add_scalar("valid/pearson_correlation", r2, epo)
                        writer.add_scalar("valid/concordance_index", CI, epo)
                        writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
                table.add_row(lst)
            else:
                model_max = copy.deepcopy(self.model)

        # load early stopped model
        self.model = model_max

        if val is not None:
            #### after training
            prettytable_file = os.path.join(
                self.result_folder, "valid_markdowntable.txt"
            )
            with open(prettytable_file, "w", encoding="utf8") as fp:
                fp.write(table.get_string())

        if test is not None:
            if verbose:
                print("--- Go for Testing ---")
            mse, r2, p_val, CI, logits, _ = self.test_(testing_generator, model_max)
            test_table = PrettyTable(
                ["MSE", "Pearson Correlation", "with p-value", "Concordance Index"]
            )
            test_table.add_row(list(map(float2str, [mse, r2, p_val, CI])))
            if verbose:
                print(
                    "Testing MSE: "
                    + str(mse)
                    + " , Pearson Correlation: "
                    + str(r2)
                    + " with p-value: "
                    + str(f"{p_val:.2E}")
                    + " , Concordance Index: "
                    + str(CI)
                )
            np.save(
                os.path.join(
                    self.result_folder,
                    str(self.drug_encoding)
                    + "_"
                    + str(self.target_encoding)
                    + "_logits.npy",
                ),
                np.array(logits),
            )

            ######### learning record ###########

            ### 1. test results
            prettytable_file = os.path.join(
                self.result_folder, "test_markdowntable.txt"
            )
            with open(prettytable_file, "w", encoding="utf8") as fp:
                fp.write(test_table.get_string())

        ### 2. learning curve
        fontsize = 16
        iter_num = list(range(1, len(loss_history) + 1))
        plt.figure(3)
        plt.plot(iter_num, loss_history, "bo-")
        plt.xlabel("iteration", fontsize=fontsize)
        plt.ylabel("loss value", fontsize=fontsize)
        pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
        with open(pkl_file, "wb") as pck:
            pickle.dump(loss_history, pck)

        fig_file = os.path.join(self.result_folder, "loss_curve.png")
        plt.savefig(fig_file)
        if verbose:
            print("--- Training Finished ---")
            writer.flush()
            writer.close()

    def predict(self, df_data):
        """
        utils.data_process_repurpose_virtual_screening
        pd.DataFrame
        """
        print("predicting...")
        info = data_process_loader(
            df_data.index.values, df_data.Label.values, df_data, **self.config
        )
        self.model.to(self.device)
        params = {
            "batch_size": self.config["batch_size"],
            "shuffle": False,
            "num_workers": self.config["num_workers"],
            "drop_last": False,
            "sampler": SequentialSampler(info),
        }
        generator = data.DataLoader(info, **params)

        score = self.test_(generator, self.model, repurposing_mode=True)
        return score

    def save_model(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(self.model.state_dict(), path_dir + "/model.pt")
        save_dict(path_dir, self.config)

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        state_dict = torch.load(path, map_location=torch.device("cpu"))
        # to support training from multi-gpus data-parallel:

        if next(iter(state_dict))[:7] == "module.":
            # the pretrained model is from data-parallel module
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

        self.binary = self.config["binary"]
