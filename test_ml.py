import pytest

# TODO: add necessary import
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics
from sklearn.model_selection import train_test_split


def data():
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    df = pd.read_csv(data_path)
    return df


# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Testing if train_model function returns RandomForestClassifier model
    """
    X = np.array([[25, 40, 1], [35, 60, 2], [45, 20, 1], [50, 30, 2]])
    y = np.array([0, 1, 1, 0])
    pass

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model():
    """
    Testing if compute_model_metrics function returns
    precision, recall, and F1 score as floats
    """
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    pass

    precision, recall, fbeta = compute_model_metrics(
        y_true, y_pred
    )

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


# TODO: implement the third test. Change the function name and input as needed
def test_size():
    """
    Testing if training and test datasets have expected sizes
    """

    df = data()
    train, test = train_test_split(df, test_size=0.2, random_state=12)

    assert len(train) + len(test) == len(df)

    test_ratio = len(test) / len(df)
    assert abs(test_ratio - 0.2) < 0.01