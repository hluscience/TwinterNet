import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import re
from sklearn.metrics import accuracy_score, roc_auc_score

class Twinter_Net(nn.Module):
    def __init__(
        self,
        X_num_features,
        Z_num_features,
        X_hidden_units,
        Z_hidden_units,
        X_Z_pairs_repeats,
        X_Z_hidden_units,
        X_Z_pairwise = True,
        X_Z_parallel = True,
        X_allZ_layer = True,
        Z_allX_layer = True,
        task_type = "regression"
    ):
        super(Twinter_Net, self).__init__()

        if not X_Z_pairwise and not X_allZ_layer and not Z_allX_layer:
            raise ValueError("When X_Z_pairwise is False, at least one of X_allZ_layer or Z_allX_layer must be True. "
                              "You have three options:"
                              "\nOption 1: Set both X_allZ_layer and Z_allX_layer to True;"
                              "\nOption 2: Set X_allZ_layer to True and Z_allX_layer to False;"
                              "\nOption 3: Set X_allZ_layer to False and Z_allX_layer to True.")

        self.X_num_features = X_num_features
        self.Z_num_features = Z_num_features
        self.X_Z_pairwise = X_Z_pairwise
        self.X_Z_parallel = X_Z_parallel
        self.X_allZ_layer = X_allZ_layer
        self.Z_allX_layer = Z_allX_layer
        self.X_Z_pairs_repeats = X_Z_pairs_repeats
        self.task_type = task_type

        # create the X net
        self.X_mlp = create_mlp([X_num_features] + X_hidden_units + [1])
        # create the Z net
        self.Z_mlp = create_mlp([Z_num_features] + Z_hidden_units + [1])
        # create the X_Z net
        X_Z_pairs = (X_num_features * Z_num_features if X_Z_pairwise else
                   X_num_features + Z_num_features if X_allZ_layer and Z_allX_layer else
                   X_num_features if X_allZ_layer else
                   Z_num_features if Z_allX_layer else 0)
        X_Z_layer_units = X_Z_pairs * X_Z_pairs_repeats
        self.X_Z_layer = nn.Linear(X_num_features + Z_num_features, X_Z_layer_units)
        self.X_Z_mask = self.create_mask(X_num_features, Z_num_features)
        with torch.no_grad():
            self.X_Z_layer.weight.mul_(self.X_Z_mask)
        self.X_Z_relu = nn.ReLU()
        if X_Z_parallel:
            self.X_Z_parallel_mlp = self.create_X_Z_nets(X_Z_pairs, X_Z_hidden_units)
        else:
            self.X_Z_mlp = create_mlp([X_Z_layer_units] + X_Z_hidden_units + [1])

    def forward(self, x, z):
        output_X = self.X_mlp(x)
        output_Z = self.Z_mlp(z)
        x_z = torch.cat((x, z), dim=1)
        x_z_layer = self.X_Z_layer(x_z)
        x_z_layer = self.X_Z_relu(self.X_Z_layer(x_z))
        if self.X_Z_parallel:
            output_X_Z = self.forward_X_Z_nets(x_z_layer, self.X_Z_parallel_mlp)
        else:
            output_X_Z = self.X_Z_mlp(x_z_layer)
        output_sum = output_X + output_Z + output_X_Z
        if self.task_type == "regression":
            return output_sum
        elif self.task_type == "classification":
            return torch.sigmoid(output_sum)

    def create_X_Z_nets(self, X_Z_pairs, X_Z_hidden_units):
        x_z_mlp_list = [
            create_mlp([self.X_Z_pairs_repeats] + X_Z_hidden_units + [1], out_bias=False)
            for i in range(X_Z_pairs)
        ]

        # X_mlp, Z_mlp, X_Z_layer always exist
        # when X_Z_parallel == False: X_Z_mlp
        # when X_Z_parallel == True:
        # 1. X_Z_pairwise == True: Xj_Zk_mlp
        # 2. X_Z_pairwise == False & X_allZ_layer = True & Z_allX_layer = True: Xj_Z_mlp, X_Zk_mlp
        # 3. X_Z_pairwise == False & X_allZ_layer = True & Z_allX_layer = False: Xj_Z_mlp
        # 4. X_Z_pairwise == False & X_allZ_layer = False & Z_allX_layer = True: X_Zk_mlp

        if self.X_Z_pairwise:
            for j in range(self.X_num_features):
                for k in range(self.Z_num_features):
                    setattr(self, "X" + str(j) + "_Z" + str(k) + "_mlp", x_z_mlp_list[j * self.Z_num_features + k])
        else:
            if self.X_allZ_layer and self.Z_allX_layer:
                for j in range(self.X_num_features):
                    setattr(self, "X" + str(j) + "_Z" + "_mlp", x_z_mlp_list[j])
                for k in range(self.Z_num_features):
                    setattr(self, "X" + "_Z" + str(k) + "_mlp", x_z_mlp_list[self.X_num_features + k])
            elif self.X_allZ_layer:
                for j in range(self.X_num_features):
                    setattr(self, "X" + str(j) + "_Z" + "_mlp", x_z_mlp_list[j])
            elif self.Z_allX_layer:
                for k in range(self.Z_num_features):
                    setattr(self, "X" + "_Z" + str(k) + "_mlp", x_z_mlp_list[k])

        return x_z_mlp_list

    def forward_X_Z_nets(self, x_z_layer, mlps):
        forwarded_x_z_mlps = []
        for i, mlp in enumerate(mlps):
            x_z_layer_selected_columns = slice(i*self.X_Z_pairs_repeats, (i+1)*self.X_Z_pairs_repeats)
            forwarded_x_z_mlps.append(mlp(x_z_layer[:, x_z_layer_selected_columns]))
        forwarded_x_z_mlp = sum(forwarded_x_z_mlps)
        return forwarded_x_z_mlp

    def create_mask(self, p, q):
        if self.X_Z_pairwise:
            vec, identity = np.ones(q), np.eye(q)
            mask_x = np.zeros((p*q, p))
            for i in range(p):
                mask_x[i*q:(i+1)*q, i] = vec
            mask_z = np.vstack([identity] * p)
            mask = np.append(mask_x, mask_z, axis=1)
        else:
            if self.X_allZ_layer and self.Z_allX_layer:
                mask = np.block([[np.eye(p), np.ones((p, q))], [np.ones((q, p)), np.eye(q)]])
            elif self.X_allZ_layer:
                mask = np.block([np.eye(p), np.ones((p, q))])
            elif self.Z_allX_layer:
                mask = np.block([np.ones((q, p)), np.eye(q)])
        mask_repeated = np.repeat(mask, repeats=self.X_Z_pairs_repeats, axis=0)

        return torch.tensor(mask_repeated, dtype=torch.float32)


def create_mlp(layer_sizes, out_bias=True):
    ls = list(layer_sizes)
    layers = nn.ModuleList()
    for i in range(1, len(ls) - 1):
        layers.append(nn.Linear(int(ls[i - 1]), int(ls[i])))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(int(ls[-2]), int(ls[-1]), bias=out_bias))
    return nn.Sequential(*layers)
    
def train(
    net,
    data_loaders,
    nepochs=100,
    verbose=False,
    early_stopping=True,
    patience=5,
    l1_const=5e-5,
    l2_const=0,
    learning_rate=1e-2,
    penalize_MMLP = False,
    opt_func=optim.Adam,
    device=torch.device("cpu"),
):

    X_num_features = net.X_num_features
    Z_num_features = net.Z_num_features
    X_Z_pairwise = net.X_Z_pairwise
    X_Z_parallel = net.X_Z_parallel
    X_allZ_layer = net.X_allZ_layer
    Z_allX_layer = net.Z_allX_layer
    mask = net.X_Z_mask.to(device)
    task_type = net.task_type

    optimizer = opt_func(net.parameters(), lr=learning_rate, weight_decay=l2_const) # The aim of weight decay is to regularize the model to prevent overfittin
    criterion = nn.MSELoss(reduction="mean") if task_type == "regression" else nn.BCELoss(reduction="mean")

    def evaluate_loss(net, data_loader, criterion, device):
        losses = []
        for X_inputs, Z_inputs, targets in data_loader:
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            loss = criterion(net(X_inputs, Z_inputs), targets).cpu().data # the loss computed by the criterion is moved from the GPU memory to the CPU memory with the .cpu() method and then converted to a plain Python number with the .data attribute.
            losses.append(loss)
        return torch.stack(losses).mean() #  Given a list of tensors, torch.stack will concatenate them to create a single tensor with size equal to the length of the list of tensors.

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
        for i, data in enumerate(data_loaders["train"], 0): #  The 0 argument specifies the starting value of the counter variable in the enumeration. By default, the counter variable starts from 0, but you can set it to start from any other integer value.
            X_inputs, Z_inputs, targets = data
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(X_inputs, Z_inputs)
            loss = criterion(outputs, targets).mean()

            reg_loss = 0
            if not X_Z_pairwise or not X_Z_parallel or penalize_MMLP:
                for name, param in net.named_parameters():
                    if name == "X_Z_layer.weight" and not X_Z_pairwise:
                        if X_allZ_layer and Z_allX_layer:
                            reg_loss += (torch.sum(torch.abs(param[:X_num_features, X_num_features:])) + torch.sum(torch.abs(param[X_num_features:, :X_num_features])))
                        elif X_allZ_layer:
                            reg_loss += torch.sum(torch.abs(param[:, X_num_features:]))
                        elif Z_allX_layer:
                            reg_loss += torch.sum(torch.abs(param[:, :X_num_features]))
                    if ("X_Z_mlp" in name and "weight" in name) and not X_Z_parallel:
                        reg_loss += torch.sum(torch.abs(param))
                    if (re.match(r"^X_mlp\.\d+\.weight$", name) or re.match(r"^Z_mlp\.\d+\.weight$", name)) and penalize_MMLP:
                        reg_loss += torch.sum(torch.abs(param))

            (loss + reg_loss * l1_const).backward()
            # mask gradients for the X_Z_layer
            with torch.no_grad():
                net.X_Z_layer.weight.grad.mul_(mask)
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
                        break

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

