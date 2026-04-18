import numpy as np
import sklearn
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, roc_auc_score

def HL_train(
    data_partition,
    task_type = "regression",
    cv_fold = 5,
):

    # Access data
    Xd, Zd, Yd = data_partition
    X_train, Z_train, Y_train = Xd['train'], Zd['train'], Yd['train']
    X_test, Z_test, Y_test = Xd['test'], Zd['test'], Yd['test']
    p = X_train.shape[1]
    q = Z_train.shape[1]

    # Define CV Lasso with main effects
    lasso_main_cv = LassoCV(cv=cv_fold, random_state=0) if task_type == "regression" else LogisticRegressionCV(cv=cv_fold, penalty='l1', solver='liblinear', random_state=0)

    # Fit model on training data
    XZ_train = np.hstack((X_train, Z_train))
    lasso_main_cv.fit(XZ_train, Y_train)

    # Split the coefficients for X and Z
    coefs = lasso_main_cv.coef_ if task_type == "regression" else lasso_main_cv.coef_[0]
    X_coefs = coefs[:p]
    Z_coefs = coefs[p:]

    # Get the indices of the non-zero features
    X_non_zero_indices = np.nonzero(X_coefs)[0]
    Z_non_zero_indices = np.nonzero(Z_coefs)[0]
    topk_X, topk_Z = len(X_non_zero_indices), len(Z_non_zero_indices)

    # Filter X and Z to only include the non-zero features
    X_non_zero_train, Z_non_zero_train = X_train[:, X_non_zero_indices], Z_train[:, Z_non_zero_indices]
    X_non_zero_test, Z_non_zero_test = X_test[:, X_non_zero_indices], Z_test[:, Z_non_zero_indices]

    # Create pairwise interactions within X, within Z, and between X and Z
    XX_interactions_train = np.array([X_non_zero_train[:, i] * X_non_zero_train[:, j] for i in range(topk_X) for j in range(i, topk_X)]).T
    ZZ_interactions_train = np.array([Z_non_zero_train[:, i] * Z_non_zero_train[:, j] for i in range(topk_Z) for j in range(i, topk_Z)]).T
    XZ_interactions_train = np.array([X_non_zero_train[:, i] * Z_non_zero_train[:, j] for i in range(topk_X) for j in range(topk_Z)]).T

    XX_interactions_test = np.array([X_non_zero_test[:, i] * X_non_zero_test[:, j] for i in range(topk_X) for j in range(i, topk_X)]).T
    ZZ_interactions_test = np.array([Z_non_zero_test[:, i] * Z_non_zero_test[:, j] for i in range(topk_Z) for j in range(i, topk_Z)]).T
    XZ_interactions_test = np.array([X_non_zero_test[:, i] * Z_non_zero_test[:, j] for i in range(topk_X) for j in range(topk_Z)]).T

    # Generate list of feature pairs (original indices) for interactions within X, within Z, and between X and Z
    XX_interaction_indices = [(X_non_zero_indices[i], X_non_zero_indices[j]) for i in range(topk_X) for j in range(i, topk_X)]
    ZZ_interaction_indices = [(Z_non_zero_indices[i], Z_non_zero_indices[j]) for i in range(topk_Z) for j in range(i, topk_Z)]
    XZ_interaction_indices = [(X_non_zero_indices[i], Z_non_zero_indices[j]) for i in range(topk_X) for j in range(topk_Z)]

    # Construt design matrix (all X, all Z, hierarchical XX, hierarchical ZZ, hierarchical XZ)
    XZ_combine_train = np.hstack((X_train, Z_train, XX_interactions_train, ZZ_interactions_train, XZ_interactions_train))
    XZ_combine_test = np.hstack((X_test, Z_test, XX_interactions_test, ZZ_interactions_test, XZ_interactions_test))

    # Define CV Lasso with hierarchical interactions
    lasso_inter_cv = LassoCV(cv=cv_fold, random_state=0) if task_type == "regression" else LogisticRegressionCV(cv=cv_fold, penalty='l1', solver='liblinear', random_state=0)

    # Fit model on training data
    lasso_inter_cv.fit(XZ_combine_train, Y_train)

    if task_type == "regression":
        # Predict on the test set
        Y_pred = lasso_inter_cv.predict(XZ_combine_test)
        # Compute the loss
        loss = mean_squared_error(Y_test, Y_pred)
        # Extract coefficients
        XZ_combine_coefs = lasso_inter_cv.coef_
    elif task_type == "classification":
        # Predict on the test set
        Y_pred = lasso_inter_cv.predict(XZ_combine_test)
        Y_pred_prob = lasso_inter_cv.predict_proba(XZ_combine_test)[:, 1]
        # Compute the loss, accuracy and auc
        loss = log_loss(Y_test, Y_pred_prob)
        accu = accuracy_score(Y_test, Y_pred)
        auc = roc_auc_score(Y_test, Y_pred_prob)
        # Extract coefficients
        XZ_combine_coefs = lasso_inter_cv.coef_[0]

    # Construct model coefficients
    XZ_combine_indexes = [(i, -1) for i in range(p)] + [(-1, j) for j in range(q)] + XX_interaction_indices + ZZ_interaction_indices + XZ_interaction_indices
    model_coefs = list(zip(XZ_combine_indexes, XZ_combine_coefs))
    model_coefs = [(tuple(np.array(i) + 1), s) for i, s in model_coefs]

    output = (model_coefs, topk_X, topk_Z, loss) if task_type == "regression" else (model_coefs, topk_X, topk_Z, loss, accu, auc)

    return output

def HL_get_interactions(model_coefs, X_num_features, Z_num_features, topk_X, topk_Z):

    p, q = X_num_features, Z_num_features
    XZ_coefs = model_coefs[-(topk_X*topk_Z):]
    XZ_index = [item[0] for item in XZ_coefs]
    XZ_strength = [np.abs(item[1]) for item in XZ_coefs]
    interaction_ranking = list(zip(XZ_index, XZ_strength)) + [((i+1, j+1), 0) for i in range(p) for j in range(q) if (i+1, j+1) not in XZ_index]
    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    interaction_ranking

    return interaction_ranking


