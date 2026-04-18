# This file is used for running simulations.
# It depends on method definitions and synthetic function definitions.
# Method Definitions:
##   TwinterNet: Defined in TwinterNet.py and BetweenView_Interaction_detection.py
##   Other Methods: Located in the "other methods" folder
##   Helper Functions: utils.py
# Synthetic function definitions: synthetic.py

import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import os
import pickle
import time
from collections import defaultdict
import re

# Diffeerent methods
def TwinterNet(
    data_partition_parm,
    model_parm,
    ground_truth,
    path
):

    # Load parameters
    X = data_partition_parm["X"]
    Z = data_partition_parm["Z"]
    Y = data_partition_parm["Y"]
    valid_size = data_partition_parm["valid_size"]
    test_size = data_partition_parm["test_size"]
    std_scale = data_partition_parm["std_scale"]
    batch_size = data_partition_parm["batch_size"]
    X_num_features = model_parm["X_num_features"]
    Z_num_features = model_parm["Z_num_features"]
    X_hidden_units = model_parm["X_hidden_units"]
    Z_hidden_units = model_parm["Z_hidden_units"]
    X_Z_pairs_repeats = model_parm["X_Z_pairs_repeats"]
    X_Z_hidden_units = model_parm["X_Z_hidden_units"]
    X_Z_pairwise = model_parm["X_Z_pairwise"]
    X_Z_parallel = model_parm["X_Z_parallel"]
    X_allZ_layer = model_parm["X_allZ_layer"]
    Z_allX_layer = model_parm["Z_allX_layer"]
    task_type = model_parm["task_type"]
    X_Z_incoming = model_parm["X_Z_incoming"]
    nepochs = model_parm["nepochs"]
    verbose = model_parm["verbose"]
    early_stopping = model_parm["early_stopping"]
    patience = model_parm["patience"]
    l1_const = model_parm["l1_const"]
    l2_const = model_parm["l2_const"]
    learning_rate = model_parm["learning_rate"]
    penalize_MMLP = model_parm["penalize_MMLP"]
    opt_func = model_parm["opt_func"]
    device = model_parm["device"]

    # Load data
    data_loaders = preprocess_data(X, Z, Y, valid_size, test_size, std_scale, batch_size)

    # Training
    start_time = time.time()
    model = Twinter_Net(X_num_features, Z_num_features, X_hidden_units, Z_hidden_units, X_Z_pairs_repeats, X_Z_hidden_units, X_Z_pairwise, X_Z_parallel, X_allZ_layer, Z_allX_layer, task_type).to(device)
    if task_type == "regression":
        model, loss = train(model, data_loaders, nepochs, verbose, early_stopping, patience, l1_const, l2_const, learning_rate, penalize_MMLP, opt_func, device)
    elif task_type == "classification":
        model, loss, accu, auc = train(model, data_loaders, nepochs, verbose, early_stopping, patience, l1_const, l2_const, learning_rate, penalize_MMLP, opt_func, device)
    end_time = time.time()
    spend_time = end_time - start_time

    # Save model
    torch.save(model.state_dict(), path)

    # Detect interactions from the weights
    model_weights = get_weights(model)
    interactions = get_interactions(model_weights, X_num_features, Z_num_features, X_Z_pairs_repeats, X_Z_incoming, X_allZ_layer, Z_allX_layer, one_indexed=True)
    aucInter = get_auc(interactions, ground_truth)
    output = (aucInter, loss, None, None, spend_time) if task_type == "regression" else (aucInter, loss, accu, auc, spend_time)

    return output
    
    
def MLP(
    data_partition_parm,
    model_parm,
    ground_truth,
    path
):
    # Load parameters
    X = data_partition_parm["X"]
    Z = data_partition_parm["Z"]
    Y = data_partition_parm["Y"]
    valid_size = data_partition_parm["valid_size"]
    test_size = data_partition_parm["test_size"]
    std_scale = data_partition_parm["std_scale"]
    batch_size = data_partition_parm["batch_size"]
    X_num_features = model_parm["X_num_features"]
    Z_num_features = model_parm["Z_num_features"]
    hidden_units = model_parm["hidden_units"]
    task_type = model_parm["task_type"]
    nepochs = model_parm["nepochs"]
    verbose = model_parm["verbose"]
    early_stopping = model_parm["early_stopping"]
    patience = model_parm["patience"]
    l1_const = model_parm["l1_const"]
    l2_const = model_parm["l2_const"]
    learning_rate = model_parm["learning_rate"]
    opt_func = model_parm["opt_func"]
    device = model_parm["device"]

    # Load data
    data_loaders = preprocess_data(X, Z, Y, valid_size, test_size, std_scale, batch_size)

    # Training
    start_time = time.time()
    model = MLP_Net(X_num_features, Z_num_features, hidden_units, task_type).to(device)
    if task_type == "regression":
        model, loss = MLP_train(model, data_loaders, nepochs, verbose, early_stopping, patience, l1_const, l2_const, learning_rate, opt_func, device)
    elif task_type == "classification":
        model, loss, accu, auc = MLP_train(model, data_loaders, nepochs, verbose, early_stopping, patience, l1_const, l2_const, learning_rate, opt_func, device)
    end_time = time.time()
    spend_time = end_time - start_time

    # Save model
    torch.save(model.state_dict(), path)

    # Detect interactions from the weights
    model_weights = MLP_get_weights(model)
    interactions = MLP_get_interactions(model_weights, X_num_features, Z_num_features, one_indexed=True)
    aucInter = get_auc(interactions, ground_truth)

    output = (aucInter, loss, None, None, spend_time) if task_type == "regression" else (aucInter, loss, accu, auc, spend_time)

    return output


def HL(
    data_partition_parm,
    model_parm,
    ground_truth,
    path
):

    # Load parameters
    X = data_partition_parm["X"]
    Z = data_partition_parm["Z"]
    Y = data_partition_parm["Y"]
    test_size = data_partition_parm["test_size"]
    std_scale = data_partition_parm["std_scale"]

    X_num_features = model_parm["X_num_features"]
    Z_num_features = model_parm["Z_num_features"]
    task_type = model_parm["task_type"]
    cv_fold = model_parm["cv_fold"]

    # Load data
    data_partition = preprocess_data2(X, Z, Y, test_size = test_size, std_scale=std_scale)

    # Training
    start_time = time.time()
    if task_type == "regression":
        model_coefs, topk_X, topk_Z, loss = HL_train(data_partition, task_type, cv_fold)
    elif task_type == "classification":
        model_coefs, topk_X, topk_Z, loss, accu, auc = HL_train(data_partition, task_type, cv_fold)
    end_time = time.time()
    spend_time = end_time - start_time

    # Save model
    with open(path, 'wb') as f:
        pickle.dump(model_coefs, f)

    # Detect interactions from the weights
    interactions = HL_get_interactions(model_coefs, X_num_features, Z_num_features, topk_X, topk_Z)
    aucInter = get_auc(interactions, ground_truth)

    output = (aucInter, loss, None, None, spend_time) if task_type == "regression" else (aucInter, loss, accu, auc, spend_time)

    return output


def EBM(
    data_partition_parm,
    model_parm,
    ground_truth,
    path
):

    # Load parameters
    X = data_partition_parm["X"]
    Z = data_partition_parm["Z"]
    Y = data_partition_parm["Y"]
    test_size = data_partition_parm["test_size"]
    std_scale = data_partition_parm["std_scale"]

    X_num_features = model_parm["X_num_features"]
    Z_num_features = model_parm["Z_num_features"]
    task_type = model_parm["task_type"]
    interactions_count = model_parm["interactions_count"]

    # Load data
    data_partition = preprocess_data2(X, Z, Y, test_size = test_size, std_scale=std_scale)

    # Training
    start_time = time.time()
    if task_type == "regression":
        ebm_global, loss = EBM_train(data_partition, task_type, interactions_count)
    elif task_type == "classification":
        ebm_global, loss, accu, auc  = EBM_train(data_partition, task_type, interactions_count)
    end_time = time.time()
    spend_time = end_time - start_time

    # Save model
    with open(path, 'wb') as f:
        pickle.dump(ebm_global, f)

    # Detect interactions from the weights
    interactions = EBM_get_interactions(ebm_global, X_num_features, Z_num_features)
    aucInter = get_auc(interactions, ground_truth)

    output = (aucInter, loss, None, None, spend_time) if task_type == "regression" else (aucInter, loss, accu, auc, spend_time)

    return output
    
    
def ParaACE(
    data_partition_parm,
    model_parm,
    ground_truth,
    path
):
    # Load parameters
    X = data_partition_parm["X"]
    Z = data_partition_parm["Z"]
    Y = data_partition_parm["Y"]
    valid_size = data_partition_parm["valid_size"]
    test_size = data_partition_parm["test_size"]
    std_scale = data_partition_parm["std_scale"]
    batch_size = data_partition_parm["batch_size"]
    X_num_features = model_parm["X_num_features"]
    Z_num_features = model_parm["Z_num_features"]
    hidden_units = model_parm["hidden_units"]
    task_type = model_parm["task_type"]
    nepochs = model_parm["nepochs"]
    verbose = model_parm["verbose"]
    early_stopping = model_parm["early_stopping"]
    patience = model_parm["patience"]
    interactions_count = model_parm["interactions_count"]
    device = model_parm["device"]

    # Load data
    Xd, Zd, Yd, data_loaders = preprocess_data3(X, Z, Y, valid_size, test_size, std_scale, batch_size)

    # Training
    start_time = time.time()
    model = Initial_FC_Net(X_num_features, Z_num_features, hidden_units, task_type).to(device)
    if task_type == "regression":
        model, loss = Initial_FC_train(model, data_loaders, nepochs, verbose, early_stopping, patience, device)
    elif task_type == "classification":
        model, loss, accu, auc = Initial_FC_train(model, data_loaders, nepochs, verbose, early_stopping, patience, device)
    end_time = time.time()
    spend_time = end_time - start_time

    # Save model
    torch.save(model.state_dict(), path)

    # Detect interactions
    X_train, Z_train, Y_train = Xd['train'], Zd['train'], Yd['train']
    interaction_strength, record_chosen_arms = detect_XZ_Hessian_UCB(model, X_train, Z_train, interactions_count, device, verbose)

    # Filter interactions
    filtered_interactions = [
        ((int(pair[0][1:]), int(pair[1][1:])), value)
        for pair, value in interaction_strength
        if pair[0].startswith('X') and pair[1].startswith('Z')
    ]
    XZ_index = [item[0] for item in filtered_interactions]
    interaction_ranking = filtered_interactions + [
        ((i + 1, j + 1), 0) for i in range(X_train.shape[1]) for j in range(Z_train.shape[1]) if (i + 1, j + 1) not in XZ_index
    ]
    interaction_ranking.sort(key=lambda x: x[1], reverse=True)

    # Compute AUC
    aucInter = get_auc(interaction_ranking, ground_truth)

    output = (aucInter, loss, None, None, spend_time) if task_type == "regression" else (aucInter, loss, accu, auc, spend_time)

    return output

def set_TwinterNet_parameter(
        X_num_features = 5,
        Z_num_features = 5,
        X_hidden_units = [100, 60, 20],
        Z_hidden_units = [100, 60, 20],
        X_Z_pairs_repeats = 10,
        X_Z_hidden_units = [10, 10],
        X_Z_pairwise = True,
        X_Z_parallel = True,
        X_allZ_layer = True,
        Z_allX_layer = True,
        task_type = "regression",
        X_Z_incoming = "mean",
        nepochs = 100,
        verbose = False,
        early_stopping = True,
        patience = 5,
        penalize_MMLP = False,
        l1_const = 5e-5,
        l2_const = 0,
        learning_rate = 1e-2,
        opt_func = optim.Adam,
        device = torch.device("cpu")
        ):

    parm = {"X_num_features": X_num_features,
            "Z_num_features": Z_num_features,
            "X_hidden_units": X_hidden_units,
            "Z_hidden_units": Z_hidden_units,
            "X_Z_pairs_repeats": X_Z_pairs_repeats,
            "X_Z_hidden_units": X_Z_hidden_units,
            "X_Z_pairwise": X_Z_pairwise,
            "X_Z_parallel": X_Z_parallel,
            "X_allZ_layer": X_allZ_layer,
            "Z_allX_layer": Z_allX_layer,
            "task_type": task_type,
            "X_Z_incoming": X_Z_incoming,
            "nepochs" : nepochs,
            "verbose" : verbose,
            "early_stopping" : early_stopping,
            "patience" : patience,
            "penalize_MMLP": penalize_MMLP,
            "l1_const" : l1_const,
            "l2_const" : l2_const,
            "learning_rate" : learning_rate,
            "opt_func" : opt_func,
            "device" : device}

    return parm


def set_MLP_parameter(
        X_num_features = 5,
        Z_num_features = 5,
        hidden_units = [100, 60, 20],
        task_type = "regression",
        nepochs = 100,
        verbose = False,
        early_stopping = True,
        patience = 5,
        l1_const = 5e-5,
        l2_const = 0,
        learning_rate = 1e-2,
        opt_func = optim.Adam,
        device = torch.device("cpu")
        ):

    parm = {"X_num_features": X_num_features,
            "Z_num_features": Z_num_features,
            "hidden_units": hidden_units,
            "task_type" : task_type,
            "nepochs" : nepochs,
            "verbose" : verbose,
            "early_stopping" : early_stopping,
            "patience" : patience,
            "l1_const" : l1_const,
            "l2_const" : l2_const,
            "learning_rate" : learning_rate,
            "opt_func" : opt_func,
            "device" : device}

    return parm

def set_HL_parameter(
        X_num_features = 5,
        Z_num_features = 5,
        task_type = "regression",
        cv_fold = 5
        ):

    parm = {"X_num_features": X_num_features,
            "Z_num_features": Z_num_features,
            "task_type": task_type,
            "cv_fold": cv_fold}

    return parm

def set_EBM_parameter(
        X_num_features = 5,
        Z_num_features = 5,
        task_type = "regression",
        interactions_count = 100
        ):

    parm = {"X_num_features": X_num_features,
            "Z_num_features": Z_num_features,
            "task_type": task_type,
            "interactions_count": interactions_count}

    return parm

def set_ParaACE_parameter(
        X_num_features = 5,
        Z_num_features = 5,
        hidden_units = [100, 60, 20],
        task_type = "regression",
        nepochs = 100,
        verbose = False,
        early_stopping = True,
        patience = 5,
        device = torch.device("cpu"),
        interactions_count = 20
        ):

    parm = {"X_num_features": X_num_features,
            "Z_num_features": Z_num_features,
            "hidden_units": hidden_units,
            "task_type" : task_type,
            "nepochs" : nepochs,
            "verbose" : verbose,
            "early_stopping" : early_stopping,
            "patience" : patience,
            "device" : device,
            "interactions_count": interactions_count}

    return parm


def set_parameter(
        root_path,
        goal,
        synth_funcs,
        models,
        task_type = "regression",
        X_num_features = 5,
        Z_num_features = 5,
        X_discrete_percentage = 0,
        Z_discrete_percentage = 0,
        X_Z_uniform = True,
        uniform_lower = -1,
        uniform_upper = 1,
        X_Z_corr = 0,
        random_error = False,
        error_scale = 0,
        std_scale = True,
        num_sim = 10,
        n = 30000,
        valid_size = 10000,
        test_size = 10000,
        batch_size = 100
        ):

    synth_funcs_body = {}
    for i in range(len(synth_funcs)):
        synth_funcs_body[synth_funcs[i].__name__] = inspect.getsource(synth_funcs[i])
    synth_funcs_body = dict(sorted(synth_funcs_body.items(), key=lambda item: item[0]))

    models_body = defaultdict(list)
    for name, model, para in models:
        models_body[name].append(inspect.getsource(model))
        models_body[name].append(para)

    parm = {"root_path": root_path,
            "goal": goal,
            "synth_funcs": synth_funcs_body,
            "models": models_body,
            "task_type": task_type,
            "X_num_features": X_num_features,
            "Z_num_features": Z_num_features,
            "X_discrete_percentage": X_discrete_percentage,
            "Z_discrete_percentage": Z_discrete_percentage,
            "X_Z_uniform": X_Z_uniform,
            "uniform_lower": uniform_lower,
            "uniform_upper": uniform_upper,
            "X_Z_corr": X_Z_corr,
            "random_error": random_error,
            "error_scale": error_scale,
            "std_scale": std_scale,
            "num_sim": num_sim,
            "n": n,
            "valid_size": valid_size,
            "test_size": test_size,
            "batch_size" : batch_size}

    parm_name = goal +'_parm.pkl'
    path = os.path.join(root_path, parm_name)
    with open(path, 'wb') as f:
        pickle.dump(parm, f)

    return parm


def generate_features(n, num_features, discrete_percentage, uniform, uniform_lower, uniform_upper, discrete_choices, discrete_probs, p_values_range):

    # Number of discrete and continuous features
    num_discrete_features = int(num_features * discrete_percentage)
    num_continuous_features = num_features - num_discrete_features

    if num_discrete_features == 0:
        return np.random.uniform(uniform_lower, uniform_upper, size=(n, num_features)) if uniform else np.random.normal(size=(n, num_features))
    elif num_continuous_features == 0:
        p_values = np.random.uniform(*p_values_range, size=num_features)
        features = np.array([np.random.choice(discrete_choices, size=n, p=discrete_probs(p)) for p in p_values])
        return features.T
    else:
        p_values = np.random.uniform(*p_values_range, size=num_discrete_features)
        discrete_features = np.array([np.random.choice(discrete_choices, size=n, p=discrete_probs(p)) for p in p_values])
        continuous_features = np.random.uniform(uniform_lower, uniform_upper, size=(num_continuous_features, n)) if uniform else np.random.normal(size=(num_continuous_features, n))
        features = np.concatenate((discrete_features, continuous_features), axis=0) # concatenate along columns
        return features.T

def run_simulation(parm, num_sim_add = 0, synth_funcs_add = [], models_add = []):

    # load parameters
    root_path = parm["root_path"]
    goal = parm["goal"]
    synth_funcs = parm["synth_funcs"]
    models = parm["models"]
    task_type = parm["task_type"]
    X_num_features = parm["X_num_features"]
    Z_num_features = parm["Z_num_features"]
    X_discrete_percentage = parm["X_discrete_percentage"]
    Z_discrete_percentage = parm["Z_discrete_percentage"]
    X_Z_uniform = parm["X_Z_uniform"]
    uniform_lower = parm["uniform_lower"]
    uniform_upper = parm["uniform_upper"]
    X_Z_corr = parm["X_Z_corr"]
    random_error = parm["random_error"]
    error_scale = parm["error_scale"]
    num_sim = parm["num_sim"]
    n = parm["n"]
    valid_size = parm["valid_size"]
    test_size = parm["test_size"]
    batch_size = parm["batch_size"]

    # Check if need to add simulations, synth_funcs and models
    add_sim = True if num_sim_add != 0 else False
    add_func = True if synth_funcs_add != [] else False
    add_models = True if models_add !=[] else False

    # load result if we need to add num_sim or synth_funcs or models
    if add_sim or add_func or add_models:
        result_name = goal +'_result.pkl'
        path = os.path.join(root_path, result_name)
        with open(path, 'rb') as f:
            result = pickle.load(f)
        AUCInters, LOSSes, ACCUs, AUCs, Times = result['AUCInter'], result['LOSS'], result['ACCU'], result['AUC'], result['Time']
    else:
        AUCInters, LOSSes, ACCUs, AUCs, Times = {}, {}, {}, {}, {}

    if add_func:
        synth_funcs_original = synth_funcs
        synth_funcs = {}
        for i in range(len(synth_funcs_add)):
            synth_funcs[synth_funcs_add[i].__name__] = inspect.getsource(synth_funcs_add[i])
        synth_funcs = dict(sorted(synth_funcs.items(), key=lambda item: item[0]))

    if add_models:
        models_original = models
        models = defaultdict(list)
        for name, model, para in models_add:
            models[name].append(inspect.getsource(model))
            models[name].append(para)
    model_name_len = len(max(list(models.keys()), key=len))

    if add_sim:
        num_sim_start = num_sim
        num_sim_end = num_sim + num_sim_add
    else:
        num_sim_start = 0
        num_sim_end = num_sim

    # start simulation
    for i, func in enumerate(synth_funcs, 0):
        # Extract and create a function object synth_func
        print(func)
        exec(synth_funcs[func])
        synth_func = locals()[func]
        aucInters, losses, accus, aucs, times = (AUCInters[func], LOSSes[func], ACCUs[func], AUCs[func], Times[func]) if add_sim or add_models else (defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list))

        for j in range(num_sim_start, num_sim_end):
            set_seed(j, device=device)
            # Generate synthetic data with ground truth interactions
            X = generate_features(n, X_num_features, X_discrete_percentage,
                                  X_Z_uniform, uniform_lower, uniform_upper,
                                  discrete_choices = [0, 1, 2],
                                  discrete_probs = lambda p: [p**2, 2*p*(1-p), (1-p)**2],
                                  p_values_range = [0.01, 0.49])
            Z = generate_features(n, Z_num_features, Z_discrete_percentage,
                                  X_Z_uniform, uniform_lower, uniform_upper,
                                  discrete_choices = [0, 1],
                                  discrete_probs = lambda p: [p, 1-p],
                                  p_values_range = [0, 1])
            Z[:,0] = X_Z_corr*X[:,0] + np.sqrt(1-X_Z_corr**2)*Z[:,0]
            if random_error:
                error = np.random.normal(loc=0, scale=error_scale, size=n)
                Z[:,0] = X[:,0] + error

            Y, ground_truth = synth_func(X, Z, task_type)
            data_partition_parm = {"X": X, "Z": Z, "Y": Y,
                                   "valid_size": valid_size, "test_size": test_size,
                                   "std_scale": True, "batch_size": batch_size}

            # Training model
            for k, model in enumerate(models, 0):
                # Create a save model path
                model_name = goal + "_" + model + "_" + func + "_sim" + str(j + 1) + ".pth"
                path = os.path.join(root_path, model_name)
                # Extract and create a function object synth_func
                exec(models[model][0])
                model_def_name = re.search(r'def (\w+)', models[model][0]).group(1) # group(0) returns def, group(1) returns "GENN"
                model_running = locals()[model_def_name]
                model_parm = models[model][1]
                # Train
                aucInter, loss, accu, auc, spend_time = model_running(data_partition_parm, model_parm, ground_truth, path)
                if task_type == "regression":
                    print("[sim %d/%d, model %s] AUCInter: %.4f, MSE: %.4f, Time: %.4f" % (j + 1, num_sim_end, "{:<{}}".format(model, model_name_len), aucInter, loss, spend_time)) #% character is a placeholder
                elif task_type == "classification":
                    print("[sim %d/%d, model %s] AUCInter: %.4f, LogLoss: %.4f, Accu: %.4f, AUC: %.4f, Time: %.4f" % (j + 1, num_sim_end, "{:<{}}".format(model, model_name_len), aucInter, loss, accu, auc, spend_time))
                aucInters[model].append(aucInter)
                losses[model].append(loss)
                accus[model].append(accu)
                aucs[model].append(auc)
                times[model].append(spend_time)

        for model in models:
            if task_type == "regression":
                print("[{:<{}} mean] AUCInter: {:.4f}, MSE: {:.4f}, Time: {:.4f}".format(model, model_name_len, np.mean(aucInters[model]), np.mean(losses[model]), np.mean(times[model])))
            elif task_type == "classification":
                print("[{:<{}} mean] AUCInter: {:.4f}, LogLoss: {:.4f}, Accu: {:.4f}, AUC: {:.4f}, Time: {:.4f}".format(model, model_name_len, np.mean(aucInters[model]), np.mean(losses[model]), np.mean(accus[model]), np.mean(aucs[model]), np.mean(times[model])))
        AUCInters[func] = aucInters
        LOSSes[func] = losses
        ACCUs[func] = accus
        AUCs[func] = aucs
        Times[func] = times
        result = {"AUCInter": aucInters, "LOSS": losses, "ACCU": accus, "AUC": aucs, "Time": times}
        result_name = goal + "_" + func + '_result.pkl'
        path = os.path.join(root_path, result_name)
        with open(path, 'wb') as f:
            pickle.dump(result, f)

    # resave parm if we need to add num_sim or synth_funcs
    if add_sim:
        parm['num_sim'] = num_sim_end
    if add_func:
        synth_funcs_original.update(synth_funcs)
        synth_funcs = dict(sorted(synth_funcs_original.items(), key=lambda item: item[0]))
        parm['synth_funcs'] = synth_funcs
        AUCInters = dict(sorted(AUCInters.items(), key=lambda item: item[0]))
        LOSSes = dict(sorted(LOSSes.items(), key=lambda item: item[0]))
        ACCUs = dict(sorted(ACCUs.items(), key=lambda item: item[0]))
        AUCs = dict(sorted(AUCs.items(), key=lambda item: item[0]))
        Times = dict(sorted(Times.items(), key=lambda item: item[0]))
    if add_models:
        models_original.update(models)
        parm['models'] = models_original
    if add_sim or add_func or add_models:
        parm_name = goal +'_parm.pkl'
        path = os.path.join(root_path, parm_name)
        with open(path, 'wb') as f:
            pickle.dump(parm, f)

    # save result
    result = {"AUCInter": AUCInters, "LOSS": LOSSes, "ACCU": ACCUs, "AUC": AUCs, "Time": Times}
    result_name = goal +'_result.pkl'
    path = os.path.join(root_path, result_name)
    with open(path, 'wb') as f:
        pickle.dump(result, f)

    return AUCInters, LOSSes, ACCUs, AUCs, Times
    
# Run experiments
# Example
root_path = F"/content/drive/My Drive/Colab Notebooks/GxE_results/new_paper_simulation"
goal = "100p20q_hier"
task_type = "regression"
X_num_features, Z_num_features = 100, 20
X_discrete_percentage, Z_discrete_percentage = 0, 0
X_Z_uniform = True
num_sim = 10
n = 10000
valid_size = 1125
test_size = 2500

synth_funcs = [synth_asym_func1_ver1, synth_asym_func2_ver2, synth_asym_func3_ver1]
models = []
models.append(("MLP", MLP, set_MLP_parameter(X_num_features, Z_num_features, hidden_units = [100, 50, 20], task_type = task_type, device = device, patience = 10, learning_rate = 5e-3)))
models.append(("TwinterNet", TwinterNet, set_TwinterNet_parameter(X_num_features, Z_num_features,  X_hidden_units = [30, 10, 5], Z_hidden_units = [8, 5, 3], X_Z_pairs_repeats = 10, X_Z_hidden_units = [10, 10], X_Z_pairwise = False, X_Z_parallel = True, X_allZ_layer = True, Z_allX_layer = False, task_type = task_type, X_Z_incoming = "min", device = device, patience = 10, learning_rate = 5e-3)))
models.append(("EBM", EBM, set_EBM_parameter(X_num_features, Z_num_features, task_type = task_type)))
models.append(("HL", HL, set_HL_parameter(X_num_features, Z_num_features, task_type = task_type)))
parm = set_parameter(root_path, goal, synth_funcs, models, task_type, X_num_features, Z_num_features, X_discrete_percentage, Z_discrete_percentage, X_Z_uniform, num_sim = num_sim, n = n, valid_size = valid_size, test_size = test_size)
AUCInters, LOSSes, ACCUs, AUCs, Times = run_simulation(parm)
