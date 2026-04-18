import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import re
from sklearn.metrics import accuracy_score, roc_auc_score
import bisect
import operator
import torch
from torch.utils import data
import re
from collections import defaultdict

class MLP_Net(nn.Module):
    def __init__(
        self,
        X_num_features,
        Z_num_features,
        hidden_units,
        task_type = "regression"
    ):
        super(MLP_Net, self).__init__()

        self.task_type = task_type
        self.interaction_mlp = create_mlp([X_num_features + Z_num_features] + hidden_units + [1])

    def forward(self, x, z):
        x_z = torch.cat((x, z), dim=1)
        output_X_Z = self.interaction_mlp(x_z)
        if self.task_type == "regression":
            return output_X_Z
        elif self.task_type == "classification":
            return torch.sigmoid(output_X_Z)
            
def MLP_train(
    net,
    data_loaders,
    nepochs=100,
    verbose=False,
    early_stopping=True,
    patience=5,
    l1_const=5e-5,
    l2_const=0,
    learning_rate=1e-2,
    opt_func=optim.Adam,
    device=torch.device("cpu"),
):

    task_type = net.task_type
    optimizer = opt_func(net.parameters(), lr=learning_rate, weight_decay=l2_const)
    criterion = nn.MSELoss(reduction="mean") if task_type == "regression" else nn.BCELoss(reduction="mean")

    def evaluate_loss(net, data_loader, criterion, device):
        losses = []
        for X_inputs, Z_inputs, targets in data_loader:
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            loss = criterion(net(X_inputs, Z_inputs), targets).cpu().data
            losses.append(loss)
        return torch.stack(losses).mean()

    def evaluate_accu(net, data_loader, device):
        accus = []
        for X_inputs, Z_inputs, targets in data_loader:
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            outputs = net(X_inputs, Z_inputs)
            accu = accuracy_score(targets.squeeze(1).detach().numpy(), outputs.squeeze(1).detach().numpy().round())
            accus.append(accu)
        return np.mean(accus)

    def evaluate_auc(net, data_loader, device):
        aucs = []
        for X_inputs, Z_inputs, targets in data_loader:
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            outputs = net(X_inputs, Z_inputs)
            auc = roc_auc_score(targets.squeeze(1).detach().cpu().numpy(), outputs.squeeze(1).detach().numpy())
            aucs.append(auc)
        return np.mean(aucs)

    best_loss = float("inf")
    best_net = None

    if "val" not in data_loaders:
        early_stopping = False

    patience_counter = 0

    if verbose:
        print("starting to train")
        if early_stopping:
            print("early stopping enabled")

    for epoch in range(nepochs):
        running_loss = 0.0
        run_count = 0
        for i, data in enumerate(data_loaders["train"], 0):
            X_inputs, Z_inputs, targets = data
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(X_inputs, Z_inputs)
            loss = criterion(outputs, targets).mean()

            reg_loss = 0

            for name, param in net.named_parameters():
                if "interaction_mlp" in name and "weight" in name:
                    reg_loss += torch.sum(torch.abs(param))
            (loss + reg_loss * l1_const).backward()
            optimizer.step()
            running_loss += loss.item()
            run_count += 1

        if epoch % 1 == 0:
            key = "val" if "val" in data_loaders else "train"
            val_loss = evaluate_loss(net, data_loaders[key], criterion, device)

            if epoch % 2 == 0:
                if verbose:
                    print(
                        "[epoch %d, total %d] train loss: %.4f, val loss: %.4f"
                        % (epoch + 1, nepochs, running_loss / run_count, val_loss)
                    )
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_net = copy.deepcopy(net)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        net = best_net
                        val_loss = best_loss
                        if verbose:
                            print("early stopping!")
                        break # The training process will stop early if the validation loss has not improved for a certain number of epochs (determined by patience).

            prev_loss = running_loss
            running_loss = 0.0

    if "test" in data_loaders:
        key = "test"
    elif "val" in data_loaders:
        key = "val"
    else:
        key = "train"

    if task_type == "regression":
        test_loss = evaluate_loss(net, data_loaders[key], criterion, device).item()
        output = (net, test_loss)
        if verbose:
            print("Finished Training. Test loss: ", test_loss)
    elif task_type == "classification":
        test_loss = evaluate_loss(net, data_loaders[key], criterion, device).item()
        test_accu = evaluate_accu(net, data_loaders[key], device)
        test_auc = evaluate_auc(net, data_loaders[key], device)
        output = (net, test_loss, test_accu, test_auc)
        if verbose:
            print("Finished Training. Test loss: %.4f, Test accuracy: %.4f, Test auc: %.4f" % (test_loss, test_accu, test_auc))

    return output

def MLP_get_weights(model):
    weights = []
    for name, param in model.named_parameters():
        if "interaction_mlp" in name and "weight" in name:
            weights.append(param.cpu().detach().numpy())
    return weights


def MLP_preprocess_weights(weights):
    w_later = np.abs(weights[-1])
    w_input = np.abs(weights[0])

    for i in range(len(weights) - 2, 0, -1):
        w_later = np.matmul(w_later, np.abs(weights[i]))

    return w_input, w_later

def MLP_interpret_interactions(w_input, w_later, X_num_features, Z_num_features):
    p = w_input.shape[1]

    interaction_ranking = []
    for i in range(X_num_features):
        for j in range(Z_num_features):
            strength = (np.minimum(w_input[:, i], w_input[:, X_num_features + j]) * w_later).sum()
            interaction_ranking.append(((i, j), strength))

    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    return interaction_ranking


def MLP_get_interactions(weights, X_num_features, Z_num_features, one_indexed=False):

    w_input, w_later = MLP_preprocess_weights(weights)

    interaction_ranking = MLP_interpret_interactions(w_input, w_later, X_num_features, Z_num_features)

    if one_indexed:
        return make_one_indexed(interaction_ranking)
    else:
        return interaction_ranking
