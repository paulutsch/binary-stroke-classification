import copy

from sklearn.model_selection import KFold

from .models import weighted_binary_cross_entropy_loss


def k_fold_cross_validation(initialized_model, X, y, k=5):
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

    model_final = copy.deepcopy(initialized_model)
    model_final.fit(X, y, X, y)

    return model_final, R_est
