import copy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split

from .evaluate import evaluate
from .utils import weighted_binary_cross_entropy_loss


def k_fold_cross_validation(initialized_model, X, y, k=5, fit_final_model=True):
    """
    Get pessimistic yet less variant empirical risk estimate using k-fold cross-validation.
    """
    kf = KFold(n_splits=k, shuffle=True)
    R_est = 0
    acc = 0
    f1 = 0
    auc = 0
    prec = 0
    rec = 0
    fpr = None
    tpr = None

    for train_index, test_index in kf.split(X):
        model_tmp = copy.deepcopy(initialized_model)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_tmp.fit(X_train, y_train, X_test, y_test)

        y_hat_proba = model_tmp.forward(X_test)
        y_hat = y_hat_proba >= 0.5
        (
            R_est_tmp,
            acc_tmp,
            f1_tmp,
            auc_tmp,
            prec_tmp,
            rec_tmp,
            cm_df,
            fpr_tmp,
            tpr_tmp,
        ) = evaluate(model_tmp.name, y_test, y_hat, y_hat_proba, plot=False)
        R_est += R_est_tmp
        acc += acc_tmp
        f1 += f1_tmp
        auc += auc_tmp
        prec += prec_tmp
        rec += rec_tmp
        fpr = fpr_tmp
        tpr = tpr_tmp

    R_est = R_est / k
    acc = acc / k
    f1 = f1 / k
    auc = auc / k
    prec = prec / k
    rec = rec / k

    if fit_final_model:
        model_final = copy.deepcopy(initialized_model)
        model_final.fit(X, y, X, y)

        return model_final, R_est, acc, f1, auc, prec, rec, cm_df, fpr, tpr

    return R_est


def feature_selection(
    X,
    y,
    n_features_per_iteration: int = 1,
    plot: bool = False,
):
    n_features = X.shape[1]
    deleted_features = []
    optimal_number_of_features_to_delete = 0

    risk_estimates = [np.inf]
    lowest_risk_estimate = np.inf

    X_tmp = X.copy()
    y_tmp = y.copy()

    original_indices = np.arange(n_features)

    for i in range(0, n_features - 1):
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.2, random_state=42
        )
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
        )
        log_reg_tmp = model.fit(X_train, y_train)
        y_pred = log_reg_tmp.predict_proba(X_val)[:, 1]
        risk_estimate = weighted_binary_cross_entropy_loss(y_pred, y_val)
        log_message = f"i = {i} / {n_features-1} – Empirical Risk Estimate: {risk_estimate}, Previous Estimate: {risk_estimates[-1]}"

        features_to_delete = np.argsort(np.abs(log_reg_tmp.coef_))[0][
            :n_features_per_iteration
        ]
        deleted_features.extend(original_indices[features_to_delete].tolist())

        X_tmp = np.delete(X_tmp, features_to_delete, axis=1)
        original_indices = np.delete(original_indices, features_to_delete)

        risk_estimates.append(risk_estimate)

        if risk_estimate < lowest_risk_estimate:
            lowest_risk_estimate = risk_estimate
            optimal_number_of_features_to_delete = i
            log_message += " – new lowest risk estimate"

        print(log_message)

    features_to_delete = deleted_features[:optimal_number_of_features_to_delete]

    if plot:
        print(
            f"Number of features to delete: {optimal_number_of_features_to_delete} / {n_features}"
        )

        plt.plot(
            range(
                0, n_features - 1 + n_features_per_iteration, n_features_per_iteration
            ),
            risk_estimates,
            label="Training Loss",
        )
        plt.xlabel("Number of deleted features")
        plt.ylabel("Weighted Binary Cross Entropy Loss")
        plt.title("Risk estimate as a function of deleted features")
        plt.legend()
        plt.show()

    return features_to_delete, risk_estimates


def nested_cross_validation(X, y, Model, param_grid, k=5):
    print(
        f"Performing nested cross-validation for model {Model.name} with {X.shape[0]} samples"
    )
    # list of one dict per parameter combination
    param_combinations = [
        dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())
    ]
    n_combinations = len(param_combinations)

    R_ests = np.zeros((k, len(param_combinations)))

    kf_outer = KFold(n_splits=k, shuffle=True)

    for i, (outer_train_idx, outer_test_idx) in enumerate(kf_outer.split(X)):
        X_train_outer, X_val_outer = X[outer_train_idx], X[outer_test_idx]
        y_train_outer, y_val_outer = y[outer_train_idx], y[outer_test_idx]

        for j, params in enumerate(param_combinations):
            model = Model(**params)

            R_ests[i, j] = k_fold_cross_validation(
                model, X_train_outer, y_train_outer, k=k, fit_final_model=False
            )
            print(
                f"Empirical risk estimate for fold {i+1} / {k}, parameter set {j+1} / {n_combinations}: {R_ests[i, j]}",
                end="\r",
            )

    R_ests_params = np.mean(R_ests, axis=0)

    params = param_combinations[np.argmin(R_ests_params)]
    print(
        f"Risk estimate for argmin(R_ests_params): {R_ests_params[np.argmin(R_ests_params)]}"
    )
    print(f"Selected best parameters: {params}")

    model_final = Model(**params)
    model_final_trained, R_est, acc, f1, auc, prec, rec, cm_df, fpr, tpr = (
        k_fold_cross_validation(model_final, X, y, k=k, fit_final_model=True)
    )
    print(
        f"{Model.name} – Empirical Risk Estimate: {R_est}, Accuracy: {acc}, F1 score: {f1}, Precision: {prec}, Recall: {rec}, AUC: {auc}\n{cm_df}"
    )

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {Model.name}")
    plt.legend(loc="lower right")
    plt.show()

    return model_final_trained, params, R_est, acc, f1, auc
