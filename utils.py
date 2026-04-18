import torch
from torch.utils import data
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def set_seed(seed=42, device=torch.device("cpu")):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

def force_float(X_numpy):
    return torch.from_numpy(X_numpy.astype(np.float32))

def convert_to_torch_loaders(Xd, Zd, Yd, batch_size):
    if type(Xd) != dict and type(Zd) != dict and type(Yd) != dict:
        Xd = {"train": Xd}
        Zd = {"train": Zd}
        Yd = {"train": Yd}

    data_loaders = {}
    for k in Xd:
        X_inputs = force_float(Xd[k]) # converts to PyTorch tensors
        Z_inputs = force_float(Zd[k])
        targets = force_float(Yd[k])
        dataset = data.TensorDataset(X_inputs, Z_inputs, targets)
        data_loaders[k] = data.DataLoader(dataset, batch_size, shuffle=(k == "train"))

    return data_loaders

def get_continuous_cols(X, unique_threshold = 10):
    continuous_cols = []
    for i in range(X.shape[1]):
        if np.issubdtype(X[:, i].dtype, np.number):  # if the column is numeric
            unique_values = np.unique(X[:, i][~np.isnan(X[:, i])])
            if len(unique_values) > unique_threshold:  # a large number of unique values
                continuous_cols.append(i)
    return continuous_cols

def standardize_data(data, continuous_cols):
    if continuous_cols:
        # Use a StandardScaler to standardize the training data, and then applies the same scaling to the validation and test data.
        scaler = StandardScaler() # StandardScaler objects
        scaler.fit(data["train"][:, continuous_cols]) # estimate the parameters of the scaling on the training data
        for k in data:
            data[k][:, continuous_cols] = scaler.transform(data[k][:, continuous_cols]) ## transform to apply the scaling to the training, validation and test data.
    return data


def preprocess_data(
    X,
    Z,
    Y,
    valid_size=500,
    test_size=500,
    std_scale=False,
    unique_threshold = 10, # used to identify discrete variables
    batch_size=100,
):

    n = X.shape[0]

    ## Make dataset splits
    ntrain, nval, ntest = n - valid_size - test_size, valid_size, test_size

    Xd = {
        "train": X[:ntrain],
        "val": X[ntrain : ntrain + nval],
        "test": X[ntrain + nval : ntrain + nval + ntest],
    }
    Zd = {
        "train": Z[:ntrain],
        "val": Z[ntrain : ntrain + nval],
        "test": Z[ntrain + nval : ntrain + nval + ntest],
    }

    Yd = {
        "train": np.expand_dims(Y[:ntrain], axis=1),
        "val": np.expand_dims(Y[ntrain : ntrain + nval], axis=1),
        "test": np.expand_dims(Y[ntrain + nval : ntrain + nval + ntest], axis=1),
    }

    # If the std_scale is TRUE, find continuous columns to standardize
    if std_scale:
        X_continuous_cols = get_continuous_cols(X, unique_threshold)
        Z_continuous_cols = get_continuous_cols(Z, unique_threshold)
        Y_continuous_cols = [0] if np.issubdtype(Y.dtype, np.number) and len(np.unique(Y)) > unique_threshold else []
        Xd = standardize_data(Xd, X_continuous_cols)
        Zd = standardize_data(Zd, Z_continuous_cols)
        Yd = standardize_data(Yd, Y_continuous_cols)

    return convert_to_torch_loaders(Xd, Zd, Yd, batch_size)


def preprocess_data2(X, Z, Y, test_size=500, std_scale=False, unique_threshold = 10):

    n = X.shape[0]

    # Make dataset splits
    ntrain = n - test_size
    Xd = {"train": X[:ntrain], "test": X[ntrain:]}
    Zd = {"train": Z[:ntrain], "test": Z[ntrain:]}
    Yd = {"train": np.expand_dims(Y[:ntrain], axis=1), "test": np.expand_dims(Y[ntrain:], axis=1)}

    # If std_scale is True, find continuous columns to standardize
    if std_scale:
        X_continuous_cols = get_continuous_cols(X, unique_threshold)
        Z_continuous_cols = get_continuous_cols(Z, unique_threshold)
        Y_continuous_cols = [0] if np.issubdtype(Y.dtype, np.number) and len(np.unique(Y)) > unique_threshold else []
        Xd = standardize_data(Xd, X_continuous_cols)
        Zd = standardize_data(Zd, Z_continuous_cols)
        Yd = standardize_data(Yd, Y_continuous_cols)
    for k in Yd:
        Yd[k] = Yd[k].flatten()

    return Xd, Zd, Yd


def preprocess_data3(
    X,
    Z,
    Y,
    valid_size=500,
    test_size=500,
    std_scale=False,
    unique_threshold = 10, # used to identify discrete variables
    batch_size=100,
):

    n = X.shape[0]

    ## Make dataset splits
    ntrain, nval, ntest = n - valid_size - test_size, valid_size, test_size

    Xd = {
        "train": X[:ntrain],
        "val": X[ntrain : ntrain + nval],
        "test": X[ntrain + nval : ntrain + nval + ntest],
    }
    Zd = {
        "train": Z[:ntrain],
        "val": Z[ntrain : ntrain + nval],
        "test": Z[ntrain + nval : ntrain + nval + ntest],
    }

    Yd = {
        "train": np.expand_dims(Y[:ntrain], axis=1),
        "val": np.expand_dims(Y[ntrain : ntrain + nval], axis=1),
        "test": np.expand_dims(Y[ntrain + nval : ntrain + nval + ntest], axis=1),
    }

    # If the std_scale is TRUE, find continuous columns to standardize
    if std_scale:
        X_continuous_cols = get_continuous_cols(X, unique_threshold)
        Z_continuous_cols = get_continuous_cols(Z, unique_threshold)
        Y_continuous_cols = [0] if np.issubdtype(Y.dtype, np.number) and len(np.unique(Y)) > unique_threshold else []
        Xd = standardize_data(Xd, X_continuous_cols)
        Zd = standardize_data(Zd, Z_continuous_cols)
        Yd = standardize_data(Yd, Y_continuous_cols)

    return Xd, Zd, Yd, convert_to_torch_loaders(Xd, Zd, Yd, batch_size)


def get_auc(interactions, ground_truth):
    strengths = []
    gt_binary_list = []
    for inter, strength in interactions:
        strengths.append(strength)
        if any(inter == gt for gt in ground_truth):
            gt_binary_list.append(1)
        else:
            gt_binary_list.append(0)

    auc = roc_auc_score(gt_binary_list, strengths)
    return auc

def print_rankings(interactions, top_k=10, spacing=14):
    print(
        justify(["Pairwise interactions"], spacing)
    )
    for i in range(top_k):
        p_inter, p_strength = interactions[i]
        print(
            justify(
                [
                    p_inter,
                    "{0:.4f}".format(p_strength),
                    ""
                ],
                spacing,
            )
        )

def justify(row, spacing=14):
    return "".join(str(item).ljust(spacing) for item in row)
