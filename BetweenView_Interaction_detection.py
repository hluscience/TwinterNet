import bisect
import operator
import numpy as np
import torch
from torch.utils import data
import re
from collections import defaultdict


def get_weights(model):

    X_Z_mlp_weights = defaultdict(list)

    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        if "X_Z_layer" in name:
            X_Z_layer_weights = param.cpu().detach().numpy()
        else:
            # when X_Z_parallel = False
            match_X_Z = re.search(r"X_Z_mlp", name)
            # when X_Z_parallel = True
            match_Xj_Zk = re.search(r"X\d+_Z\d+_mlp", name) # X_Z_pairwise = True
            match_Xj_Z = re.search(r"X\d+_Z+_mlp", name) # X_Z_pairwise = False, X_allZ_layer = True, Z_allX_layer = False
            match_X_Zk = re.search(r"X+_Z\d+_mlp", name) # X_Z_pairwise = False, X_allZ_layer = False, Z_allX_layer = True
            if match_Xj_Zk:
                X_Z_mlp_weights[match_Xj_Zk.group(0)].append(param.cpu().detach().numpy())
            elif match_Xj_Z:
                X_Z_mlp_weights[match_Xj_Z.group(0)].append(param.cpu().detach().numpy())
            elif match_X_Zk:
                X_Z_mlp_weights[match_X_Zk.group(0)].append(param.cpu().detach().numpy())
            elif match_X_Z:
                X_Z_mlp_weights[match_X_Z.group(0)].append(param.cpu().detach().numpy())

    return X_Z_layer_weights, X_Z_mlp_weights

def preprocess_weights(weights):
    X_Z_input_weights, X_Z_later_weights = weights
    w_input = np.abs(X_Z_input_weights)
    w_later = {}
    for name in X_Z_later_weights:
        mlp_weights = X_Z_later_weights[name]
        later_weights = np.abs(mlp_weights[-1])
        for i in range(len(mlp_weights) - 2, -1, -1):
            later_weights = np.matmul(later_weights, np.abs(mlp_weights[i]))
        w_later[name] = later_weights

    return w_input, w_later

def interpret_interactions(w_input, w_later, X_num_features, Z_num_features, X_Z_incoming, X_allZ_layer, Z_allX_layer, X_Z_pairs_repeats):

    w_later_list = []
    for name, value in w_later.items():
        if len(value.shape) == 2:
            value = value.flatten()
        w_later_list.extend(value.tolist())

    X_Z_pairwise = len(w_later_list) == X_num_features * Z_num_features * X_Z_pairs_repeats
    X_w_input, Z_w_input = w_input[:, :X_num_features], w_input[:, X_num_features:]
    x_w_index, z_w_index = np.arange(X_num_features), np.arange(Z_num_features)

    if X_Z_pairwise:
        X_index, Z_index = np.repeat(x_w_index, Z_num_features * X_Z_pairs_repeats), np.tile(np.repeat(z_w_index, X_Z_pairs_repeats), X_num_features)
        row_index = np.arange(w_input.shape[0])
    else:
        X_index_part1, X_index_part2 = np.repeat(x_w_index, Z_num_features * X_Z_pairs_repeats), np.tile(x_w_index, Z_num_features * X_Z_pairs_repeats)
        Z_index_part1, Z_index_part2 = np.tile(z_w_index, X_num_features * X_Z_pairs_repeats), np.repeat(z_w_index, X_num_features * X_Z_pairs_repeats)
        row_index_part1, row_index_part2 = np.repeat(np.arange(X_num_features * X_Z_pairs_repeats), Z_num_features), np.repeat(np.arange(Z_num_features * X_Z_pairs_repeats), X_num_features)

        if X_allZ_layer and Z_allX_layer:
            X_index = np.concatenate((X_index_part1, X_index_part2))
            Z_index = np.concatenate((Z_index_part1, Z_index_part2))
            row_index = np.concatenate((row_index_part1, row_index_part2 + X_num_features * X_Z_pairs_repeats))
            w_later_list1, w_later_list2 = np.repeat(w_later_list[:X_num_features*X_Z_pairs_repeats], Z_num_features), np.repeat(w_later_list[X_num_features*X_Z_pairs_repeats:], X_num_features)
            w_later_list = np.concatenate((w_later_list1, w_later_list2))
        elif X_allZ_layer:
            X_index, Z_index, row_index = X_index_part1, Z_index_part1, row_index_part1
            w_later_list = np.repeat(w_later_list, Z_num_features)
        elif Z_allX_layer:
            X_index, Z_index, row_index = X_index_part2, Z_index_part2, row_index_part2
            w_later_list = np.repeat(w_later_list, X_num_features)

    if X_Z_incoming == "mean":
        strength = np.mean([X_w_input[row_index, X_index], Z_w_input[row_index, Z_index]], axis=0) * w_later_list
    elif X_Z_incoming == "min":
        strength = np.min([X_w_input[row_index, X_index], Z_w_input[row_index, Z_index]], axis=0) * w_later_list
    interaction_strength = list(zip(zip(X_index, Z_index), strength))

    if X_Z_pairwise:
        interaction_ranking = interaction_strength
    else:
        interaction_ranking = defaultdict(int)
        for i in range(len(interaction_strength)):
            name = (X_index[i], Z_index[i])
            value = strength[i]
            interaction_ranking[name] += value
        interaction_ranking = [(name, strength) for name, strength in interaction_ranking.items()]

    interaction_ranking.sort(key=lambda x: x[1], reverse=True)

    return interaction_ranking

def make_one_indexed(interaction_ranking):
    return [(tuple(np.array(i) + 1), s) for i, s in interaction_ranking]

def get_interactions(weights, X_num_features, Z_num_features, X_Z_pairs_repeats, X_Z_incoming = "mean", X_allZ_layer = True, Z_allX_layer = True, one_indexed=False):

    w_input, w_later = preprocess_weights(weights)

    interaction_ranking = interpret_interactions(w_input, w_later, X_num_features, Z_num_features, X_Z_incoming, X_allZ_layer, Z_allX_layer, X_Z_pairs_repeats)

    if one_indexed:
        return make_one_indexed(interaction_ranking)
    else:
        return interaction_ranking
