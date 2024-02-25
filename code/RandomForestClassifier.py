import logging
from typing import List, Tuple

import numpy as np
from DecisionTreeClassifier import DecisionTreeClassifier
from scipy import stats

logging.getLogger("DecisionTreeClassifier").propagate = False


class RandomForestClassifier:
    """
    A custom implementation of the RandomForestClassifier.

    This class implements a basic version of the RandomForest algorithm for classification tasks. It builds a specified number of decision trees, each trained on a bootstrap sample of the input data. For predictions, it aggregates the predictions of all trees using majority voting. It also calculates an Out-of-Bag (OOB) score as an estimate of the model's performance.

    Parameters
    ---------

    - n_estimators (int): The number of trees in the forest. Default is 100.
    - max_depth (int, optional): The maximum depth of the tree. If None, the tree will grow until all leaves are pure or until it reaches the minimum samples split.
    - min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
    - max_features (int, optional): The number of features to consider when looking for the best split. If None, all features are considered.    - min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default is 0.0.
    - random_state (int): Controls the randomness of the bootstrapping of the samples and the features considered at each split. Default is 42.
    - debug (bool): If True, the logging level will be set to DEBUG, providing more detailed logging information. Default is False.

    Attributes
    ---------
    - oob_score_ (float): The Out-of-Bag score, an estimate of the model's performance on unseen data.
    - trees (List[DecisionTreeClassifier]): The list of fitted decision trees within the forest.

    Methods
    -------
    - fit(X, y): Fits the RandomForestClassifier to the input data X and target labels y.
    - predict(X): Predicts class labels for the input data X.
    - _bootstrap_sample(X, y): Generates a bootstrap sample from the input data and identifies Out-of-Bag indices.
    - _calculate_oob_score(oob_votes, y): Calculates the Out-of-Bag score based on the aggregated OOB predictions.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        max_features: int = None,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
        debug: bool = False,
    ) -> None:
        """
        Initializes the RandomForestClassifier with the given parameters.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease

        self.debug = debug
        self.oob_score_: float = None

        self.random_state = random_state
        self.random = np.random.RandomState(random_state)

        self._logger = logging.getLogger("RandomForestClassifier")
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.trees: List[DecisionTreeClassifier] = []

    def __repr__(self) -> str:
        return f"RandomForestClassifier(n_estimators={self.n_estimators}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_features={self.max_features})"

    def _bootstrap_sample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a bootstrap sample from the input data and identifies Out-of-Bag (OOB) indices.

        Parameters:
        - X (np.ndarray): The input features array with shape (n_samples, n_features).
        - y (np.ndarray): The target values array with shape (n_samples,).

        Returns:
        - Tuple[np.ndarray, np.ndarray]: A tuple containing the indices of the bootstrap sample and the indices of the OOB samples.
        """

        n_samples, n_features = X.shape
        bootstrap_indices = self.random.choice(n_samples, size=n_samples, replace=True)

        oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)

        return bootstrap_indices, oob_indices

    def _calculate_oob_score(self, oob_votes: np.ndarray, y: np.ndarray) -> None:
        """
        Calculates the Out-of-Bag (OOB) score based on the aggregated OOB predictions.

        The OOB score is an estimate of the model's performance on unseen data, calculated as the accuracy of predictions for samples that were not included in the bootstrap sample for each tree.

        Parameters:
        - oob_votes (np.ndarray): An array with shape (n_samples, n_classes) containing the vote count for each class, for each sample.
        - y (np.ndarray): The true target values array with shape (n_samples,).

        Updates:
        - self.oob_score_ (float): The calculated OOB score is stored in this attribute.
        """

        oob_samples_received_vote = np.sum(oob_votes, axis=1) > 0
        if np.any(oob_samples_received_vote):
            oob_predictions = np.argmax(oob_votes[oob_samples_received_vote], axis=1)
            self.oob_score_ = np.mean(oob_predictions == y[oob_samples_received_vote])

            self._logger.debug(f"OOB Score: {self.oob_score_}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the RandomForestClassifier to the input data X and target labels y.

        This method trains `n_estimators` decision trees on bootstrap samples of the input data.
        It also calculates the Out-of-Bag (OOB) score as an estimate of the model's performance.

        Parameters:
        - X (np.ndarray): The input features array with shape (n_samples, n_features).
        - y (np.ndarray): The target values array with shape (n_samples,).
        """

        n_samples, n_features = X.shape
        self.max_features = self.max_features or int(np.sqrt(n_features))

        oob_votes = np.zeros((n_samples, np.max(y) + 1))

        self._logger.debug(f"Fitting {self.n_estimators} trees...")
        for i in range(self.n_estimators):
            self._logger.debug(f"Fitting tree {i + 1}...")
            bootstrap_indices, oob_indices = self._bootstrap_sample(X, y)

            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                debug=self.debug,
            )

            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)
            self._logger.debug(f"Tree {i + 1} fitted...")

            # Collect OOB predictions for each tree
            if len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                for idx, pred in zip(oob_indices, oob_pred):
                    oob_votes[idx, pred] += 1

        self._calculate_oob_score(oob_votes, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the input data X using the trained RandomForestClassifier.

        This method aggregates the predictions of all trees in the forest using majority voting to determine the final class labels.

        Parameters:
        - X (np.ndarray): The input features array to predict, with shape (n_samples, n_features).

        Returns:
        - np.ndarray: The predicted class labels for each sample in X, with shape (n_samples,).
        """

        self._logger.debug(f"Making predictions using {self.n_estimators} trees...")

        predictions = np.array([tree.predict(X) for tree in self.trees])

        prediction = stats.mode(predictions)[0].flatten()

        return prediction
