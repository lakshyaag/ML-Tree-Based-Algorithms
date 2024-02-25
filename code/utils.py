import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def print_evaluation_metrics(
    y_true: list, y_pred: list, model, name: str
) -> Dict[str, Any]:
    """
    Prints and plots evaluation metrics for a given model's predictions.

    Parameters
    ---------
    - y_true (list): The true labels.
    - y_pred (list): The predicted labels by the model.
    - model: The model used for prediction.
    - name (str): The name of the model.

    Returns
    -------
    - Dict[str, Any]: A dictionary containing the model's name, the model object, accuracy, precision, recall, F1 score, and confusion matrix.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred)

    print(f"Model: {name}")
    print(model)

    if hasattr(model, "oob_score_"):
        print(f"OOB Score: {model.oob_score_}")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    print(f"Classification Report: \n{classification_report(y_true, y_pred)}")

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
