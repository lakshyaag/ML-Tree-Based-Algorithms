import numpy as np
import logging
from typing import List, Optional
from DecisionTreeClassifier import DecisionTreeClassifier


class GradientBoostingClassifier:
    """
    A simple implementation of a gradient boosting classifier.

    This class represents a gradient boosting model for binary classification tasks.
    It supports basic functionality such as fitting to a dataset and predicting labels for new data.

    Parameters
    ----------
    - n_estimators (int): The number of trees to build. Default is 100.
    - learning_rate (float): The learning rate controls the contribution of each tree. Default is 0.1.
    - max_depth (int, optional): The maximum depth of each tree. If None, the trees will grow until all leaves are pure.
    - min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
    - max_features (int, optional): The number of features to consider when looking for the best split. If None, all features are considered.
    - min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default is 0.0.
    - random_state (int): Controls the randomness of the bootstrapping of the samples used when building trees. Default is None.
    - debug (bool): If True, the logging level will be set to DEBUG, providing more detailed logging information. Default is False.

    Attributes
    ----------
    - trees (List[DecisionTreeClassifier]): The list of fitted trees.
    - f0 (float): The initial prediction of the model.

    Methods
    -------
    - fit(X, y): Fits the gradient boosting model to the given dataset.
    - predict(X): Predicts the class labels for the given dataset.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """
        Initializes the GradientBoostingClassifier with the specified parameters.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease

        self.random_state = random_state
        self.debug = debug

        self.trees: List[DecisionTreeClassifier] = []
        self.f0: Optional[float] = None

        self.random = np.random.RandomState(self.random_state)

        self._logger: logging.Logger = logging.getLogger("GradientBoostingClassifier")
        self._logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    def __repr__(self) -> str:
        return (
            "GradientBoostingClassifier("
            f"n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"max_features={self.max_features}, "
            f"min_impurity_decrease={self.min_impurity_decrease}, "
            f"random_state={self.random_state}"
            ")"
        )
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        The sigmoid function applied element-wise to the input array.

        Parameters:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The transformed array with the sigmoid function applied.
        """
        return 1 / (1 + np.exp(-x))

    def _log_loss(self, y: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculates the logistic loss between true labels and predictions.

        Parameters:
            y (np.ndarray): The true labels.
            predictions (np.ndarray): The predicted probabilities.

        Returns:
            float: The calculated logistic loss.
        """
        predictions = np.clip(predictions, -1e15, 1e15)

        probs = self._sigmoid(predictions)

        return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    def _gradient(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient of the logistic loss with respect to the predictions.

        Parameters:
            y (np.ndarray): The true labels.
            predictions (np.ndarray): The current predictions.

        Returns:
            np.ndarray: The gradient of the loss.
        """
        predictions = np.clip(predictions, -1e15, 1e15)

        probs = self._sigmoid(predictions)

        return -(y - probs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the gradient boosting model to the given dataset.

        Parameters:
            X (np.ndarray): The input feature array.
            y (np.ndarray): The target labels array.
        """
        self._logger.debug(f"Training tree {len(self.trees) + 1}...")

        p: float = np.mean(y)
        self.f0 = np.log(p / (1 - p)) if p > 0 and p < 1 else 0

        predictions: np.ndarray = np.full(y.shape, self.f0, dtype=np.float64)

        for _ in range(self.n_estimators):
            self._logger.debug(f"Fitting tree {_ + 1}...")

            residuals: np.ndarray = -self._gradient(y, predictions)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                debug=self.debug,
                is_regression=True,
            )

            tree.fit(X, residuals)
            self.trees.append(tree)

            predictions += self.learning_rate * tree.predict(X)

            loss: float = self._log_loss(y, predictions)
            self._logger.debug(f"Loss after fitting tree {_ + 1}: {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for the given dataset.

        Parameters:
            X (np.ndarray): The input feature array.

        Returns:
            np.ndarray: The predicted class labels.
        """
        predictions: np.ndarray = np.full(X.shape[0], self.f0, dtype=np.float64)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        probs: np.ndarray = self._sigmoid(predictions)

        return np.where(probs >= 0.5, 1, 0)
