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

    for train_index, test_index in kf.split(X):
        model_tmp = copy.deepcopy(initialized_model)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_tmp.fit(X_train, y_train, X_test, y_test)

        y_hat = model_tmp.forward(X_test)
        R_est += weighted_binary_cross_entropy_loss(y_hat, y_test)

    R_est = R_est / k

    if fit_final_model:
        model_final = copy.deepcopy(initialized_model)
        model_final.fit(X, y, X, y)

        return model_final, R_est

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
                f"Empirical risk estimate for fold {i+1} / {k}, parameter set {j+1} / {n_combinations}: {R_ests[i, j]}"
            )

    R_ests_params = np.mean(R_ests, axis=0)

    params = param_combinations[np.argmin(R_ests_params)]
    print(
        f"Risk estimate for argmin(R_ests_params): {R_ests_params[np.argmin(R_ests_params)]}"
    )
    print(f"Selected best parameters: {params}")

    model_est = Model(**params)
    model_est.fit(X_train_outer, y_train_outer, X_val_outer, y_val_outer)

    y_hat_outer_proba = model_est.forward(X_val_outer)
    y_hat_outer = model_est.predict(X_val_outer)
    R_est, acc, f1, auc = evaluate(
        Model.name, y_val_outer, y_hat_outer, y_hat_outer_proba, plot=True
    )
    print(f"Empirical risk estimate on outer validation set: {R_est}")

    model_final = Model(**params)
    model_final.fit(X, y, X, y, plot=True)
    print("Fitted final model on entire dataset")

    return model_final, params, R_est, acc, f1, auc
