import logging
from typing import List

import numpy as np
from DecisionTreeClassifier import DecisionTreeClassifier
from scipy import stats


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        min_impurity_decrease=0.0,
        random_state=42,
        debug=False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease

        self.debug = debug
        self.oob_score_ = None

        self.random_state = random_state
        self.random = np.random.RandomState(random_state)

        self._logger = logging.getLogger("RandomForestClassifier")
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.trees: List[DecisionTreeClassifier] = []

    def __repr__(self):
        return f"RandomForestClassifier(n_estimators={self.n_estimators}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_features={self.max_features})"

    def _bootstrap_sample(self, X, y):
        n_samples, n_features = X.shape
        bootstrap_indices = self.random.choice(n_samples, size=n_samples, replace=True)

        oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)

        return (
            bootstrap_indices,
            oob_indices,
        )

    def _calculate_oob_score(self, oob_votes, y):
        oob_samples_received_vote = np.sum(oob_votes, axis=1) > 0
        if np.any(oob_samples_received_vote):
            oob_predictions = np.argmax(oob_votes[oob_samples_received_vote], axis=1)
            self.oob_score_ = np.mean(oob_predictions == y[oob_samples_received_vote])

            self._logger.debug(f"OOB Score: {self.oob_score_}")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.max_features = self.max_features or int(np.sqrt(n_features))

        oob_votes = np.zeros((n_samples, np.max(y) + 1))

        self._logger.debug(f"Fitting {self.n_estimators} trees...")
        for _ in range(self.n_estimators):
            self._logger.debug(f"Fitting tree {_ + 1}...")
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
            self._logger.debug(f"Tree {_ + 1} fitted...")

            # Collect OOB predictions for each tree
            if len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                for idx, pred in zip(oob_indices, oob_pred):
                    oob_votes[idx, pred] += 1

        self._calculate_oob_score(oob_votes, y)

    def predict(self, X):
        self._logger.debug(f"Making predictions using {self.n_estimators} trees...")

        predictions = np.array([tree.predict(X) for tree in self.trees])

        prediction = stats.mode(predictions)[0].flatten()

        return prediction
