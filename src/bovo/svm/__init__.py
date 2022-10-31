import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

x_columns = [
    "Airesecteur1",
    "Airesecteur2",
    "Airesecteur3",
    "Airesecteur4",
    "Airesecteur5",
    "Airesecteur6",
    "Airesecteur7",
    "Airesecteur8",
    "SEpCo25",
    "SEpCo50",
    "SEpCo75",
    "SEpCoM",
    "SEpCo2mmC",
    "SEpTr25",
    "SEpTr50",
    "SEpTr75",
    "SEpTrM",
    "BC",
    "MC",
    "MB",
    "SLaTaOs2mmC",
    "SLaTaOs25",
    "SLaTaOs50",
    "SLaTaOs75",
    "SLaTaOsTangM",
    "SPeBaTr",
    "SPeToCoAl",
    "SPeCoBas",
    "SPeMaBas",
    "SPeToCo",
    "SPeToMa",
    "SPeToTr",
    "SPeToTrAl",
    "SSuToCoAl",
    "SSuCoBa",
    "SSuMaPaAl",
    "SSuMaPaBas",
    "SSuToCo",
    "SSuToMa",
    "SSuToTr",
    "SSuToTrPaAl",
    "SSuTrPaBas",
]


def load_data(y_col):
    df = pd.read_csv("/home/oscar/bovo/data/svm/196_exams.csv", sep=";")
    df = df[df["Cote"] == "dente"]

    x_df = df[df.columns.intersection(x_columns)]

    return x_df, np.array(df[y_col])


def load_with_threshold(y_col, loss_threshold):
    x_df, target_values = load_data(y_col)
    X = x_df.to_numpy()
    X = (X - np.mean(X, axis=0, keepdims=True)) / (
        np.var(X, axis=0, keepdims=True) + 1e-3
    )
    y = np.array([0 if v < loss_threshold else 1 for v in target_values])
    return X, y, x_df.columns


def threshold_id(thresholds, v):
    for i, t in enumerate(thresholds):
        if v < t:
            return i
    return len(thresholds)


def load_with_thresholds(y_col, thresholds):
    x_df, target_values = load_data(y_col)
    X = x_df.to_numpy()
    X = (X - np.mean(X, axis=0, keepdims=True)) / (
        np.var(X, axis=0, keepdims=True) + 1e-3
    )

    thresholds = sorted(thresholds)
    y = np.array([threshold_id(thresholds, v) for v in target_values])
    return X, y, x_df.columns


# def load_with_two_hreshold(y_col, th_min, th_max):
#     x_df, target_values = load_data(y_col)
#     selector = np.array([x < th_min or  th_max < x for x in y_col])

#     X_df = X_df[selector]
#     target_values = target_values[selector]

#     X = x_df.to_numpy()
#     X = (X - np.mean(X, axis=0, keepdims=True)) / (
#         np.var(X, axis=0, keepdims=True) + 1e-3
#     )
#     y = np.array([0 if v < loss_threshold else 1 for v in target_values])
#     return X, y, x_df.columns


def calc_score(svc, cv, X, y):
    to_return = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        _ = svc.fit(X_train, y_train)
        to_return.append(svc.score(X_test, y_test))
    return np.mean(to_return)


def select_features(svc, cv, X, y):
    scores = []
    columns_selectors = []
    for i in range(X.shape[1]):
        rfecv = RFECV(
            estimator=svc,
            step=1,
            cv=cv,
            scoring="accuracy",
            min_features_to_select=X.shape[1] - i,
        )
        rfecv.fit(X, y)
        columns_selectors.append(rfecv.support_.ravel())
        scores.append(calc_score(svc, cv, X.T[rfecv.support_.ravel()].T, y))

    return scores, columns_selectors


def select_lasso_features(cv, X, y):
    lasso = LassoCV(cv=cv)
    lasso.fit(X, y)
    return np.abs(lasso.coef_) > 1e-5


def select_lasso2_features(cv, X, y):
    lasso = LogisticRegression(penalty="l1", solver="liblinear")
    lasso.fit(X, y)
    return np.abs(lasso.coef_[0]) > 1e-5


def calc_confusion_matrix(svc, cv, loss_threshold, x, y):
    actual_classes = []
    predicted_classes = []

    all_matrixes = []
    all_acc = []

    for train_index, test_index in cv.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        _ = svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)

        all_matrixes.append(confusion_matrix(y_test, y_pred))
        all_acc.append(np.mean([x == y for x, y in zip(y_test, y_pred)]))

        actual_classes = actual_classes + list(y_test)
        predicted_classes = predicted_classes + list(y_pred)

    sorted_labels = ["< {}".format(loss_threshold), ">= {}".format(loss_threshold)]
    matrix = confusion_matrix(actual_classes, predicted_classes)
    std = np.std(all_matrixes, axis=0) * len(all_matrixes)

    texts = np.array(
        ["{} (±{:.2f})".format(m, s) for m, s in zip(matrix.flatten(), std.flatten())]
    ).reshape(matrix.shape)

    plt.figure(figsize=(12.8, 6))
    # sns.heatmap(
    #     matrix,
    #     annot=texts,
    #     xticklabels=sorted_labels,
    #     yticklabels=sorted_labels,
    #     cmap="Blues",
    #     fmt="",
    # )
    sns.heatmap(
        matrix,
        annot=True,
        xticklabels=sorted_labels,
        yticklabels=sorted_labels,
        cmap="Blues",
        fmt="g",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    return np.mean(all_acc), np.std(all_acc)


def calc_multi_confusion_matrix(svc, cv, texts, x, y):
    actual_classes = []
    predicted_classes = []

    all_matrixes = []
    all_acc = []

    for train_index, test_index in cv.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        _ = svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)

        all_matrixes.append(confusion_matrix(y_test, y_pred))
        all_acc.append(np.mean([x == y for x, y in zip(y_test, y_pred)]))

        actual_classes = actual_classes + list(y_test)
        predicted_classes = predicted_classes + list(y_pred)

    matrix = confusion_matrix(actual_classes, predicted_classes)
    std = np.std(all_matrixes, axis=0) * len(all_matrixes)

    # texts = np.array(
    #     ["{} (±{:.2f})".format(m, s) for m, s in zip(matrix.flatten(), std.flatten())]
    # ).reshape(matrix.shape)

    plt.figure(figsize=(12.8, 6))
    # sns.heatmap(
    #     matrix,
    #     annot=texts,
    #     xticklabels=sorted_labels,
    #     yticklabels=sorted_labels,
    #     cmap="Blues",
    #     fmt="",
    # )
    sns.heatmap(
        matrix,
        annot=True,
        xticklabels=texts,
        yticklabels=texts,
        cmap="Blues",
        fmt="g",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    return np.mean(all_acc), np.std(all_acc)
