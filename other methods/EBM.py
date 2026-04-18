from interpret.glassbox import ExplainableBoostingRegressor
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, roc_auc_score

def EBM_train(
    data_partition,
    task_type = "regression",
    interactions_count = 10
):

    # Access data
    Xd, Zd, Yd = data_partition
    X_train, Z_train, Y_train = Xd['train'], Zd['train'], Yd['train']
    X_test, Z_test, Y_test = Xd['test'], Zd['test'], Yd['test']
    p = X_train.shape[1]
    q = Z_train.shape[1]

    # Construct design matrix
    XZ_train = np.hstack((X_train, Z_train))
    XZ_test = np.hstack((X_test, Z_test))
    X_col_names = ['X'+str(i+1) for i in range(p)]
    Z_col_names = ['Z'+str(j+1) for j in range(q)]
    col_names = X_col_names + Z_col_names
    XZ_train_df = pd.DataFrame(XZ_train, columns=col_names)
    XZ_test_df = pd.DataFrame(XZ_test, columns=col_names)

    # Train EBM
    ebm = ExplainableBoostingRegressor(interactions=interactions_count, random_state=42) if task_type == "regression" else ExplainableBoostingClassifier(interactions=interactions_count, random_state=42)
    ebm.fit(XZ_train_df, Y_train)
    ebm_global = ebm.explain_global()

    if task_type == "regression":
        # Predict on the test set
        Y_pred = ebm.predict(XZ_test_df)
        # Compute the loss
        loss = mean_squared_error(Y_test, Y_pred)
    elif task_type == "classification":
        # Predict on the test set
        Y_pred = ebm.predict(XZ_test_df)
        Y_pred_prob = ebm.predict_proba(XZ_test_df)[:, 1]
        # Compute the loss, accuracy and auc
        loss = log_loss(Y_test, Y_pred_prob)
        accu = accuracy_score(Y_test, Y_pred)
        auc = roc_auc_score(Y_test, Y_pred_prob)

    output = (ebm_global, loss) if task_type == "regression" else (ebm_global, loss, accu, auc)

    return output


def EBM_get_interactions(ebm_global, X_num_features, Z_num_features):

    p, q = X_num_features, Z_num_features
    index = ebm_global._internal_obj['overall']['names']
    strength = ebm_global._internal_obj['overall']['scores']
    model_coefs = list(zip(index, strength)) # list of tuple
    XZ_strength = [coef for coef in model_coefs if 'X' in coef[0] and 'Z' in coef[0]]
    XZ_strength = [(tuple(int(num[1:]) for num in pair.split(' & ')), strength) for pair, strength in XZ_strength]
    XZ_index = [item[0] for item in XZ_strength]
    interaction_ranking = XZ_strength + [((i+1, j+1), 0) for i in range(p) for j in range(q) if (i+1, j+1) not in XZ_index]
    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    interaction_ranking

    return interaction_ranking
