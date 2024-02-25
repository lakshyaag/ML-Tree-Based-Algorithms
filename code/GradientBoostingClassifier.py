import numpy as np
import logging

from DecisionTreeClassifier import DecisionTreeClassifier


class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        min_impurity_decrease=0.0,
        random_state=None,
        debug=False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease

        self.random_state = random_state
        self.debug = debug
        self.trees = []

        self.f0 = None

        self.random = np.random.RandomState(self.random_state)

        self._logger = logging.getLogger("GradientBoostingClassifier")
        self._logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    def __repr__(self):
        return f"GradientBoostingClassifier(n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_features={self.max_features}, min_impurity_decrease={self.min_impurity_decrease}, random_state={self.random_state})"

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_loss(self, y, predictions):
        predictions = np.clip(predictions, -1e15, 1e15)

        probs = self._sigmoid(predictions)

        return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    def _gradient(self, y, predictions):
        predictions = np.clip(predictions, -1e15, 1e15)

        probs = self._sigmoid(predictions)

        return -(y - probs)

    def fit(self, X, y):
        self._logger.debug(f"Training tree {len(self.trees) + 1}...")

        p = np.mean(y)
        self.f0 = np.log(p / (1 - p)) if p > 0 and p < 1 else 0

        predictions = np.full(y.shape, self.f0, dtype=np.float64)

        for _ in range(self.n_estimators):
            self._logger.debug(f"Fitting tree {_ + 1}...")

            residuals = -self._gradient(y, predictions)

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

            loss = self._log_loss(y, predictions)
            self._logger.debug(f"Loss after fitting tree {_ + 1}: {loss:.4f}")

    def predict(self, X):
        predictions = np.full(X.shape[0], self.f0, dtype=np.float64)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        probs = self._sigmoid(predictions)

        return np.where(probs >= 0.5, 1, 0)
