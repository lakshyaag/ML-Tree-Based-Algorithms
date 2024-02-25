import logging
import numpy as np
from graphviz import Digraph
from Node import Node
from typing import Tuple, List, Union


class DecisionTreeClassifier:
    """
    A class used to represent a decision tree classifier.

    Attributes:

        max_depth (int): The maximum depth of the tree. If None, the tree will grow until all leaves are pure.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        max_features (int): The number of features to consider when looking for the best split. If None, all features are considered.
        debug (bool): If True, the logging level will be set to DEBUG, providing more detailed logging information.
        random (np.random.RandomState): A random number generator.
        root (Node): The root node of the decision tree.
        _logger (logging.Logger): The logger for the decision tree classifier.
    """

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        max_features: int = None,
        random_state: int = 42,
        debug: bool = False,
    ) -> None:
        """
        Initializes the DecisionTreeClassifier with the given parameters.

        Parameters:
            max_depth (int): The maximum depth of the tree. If None, the tree will grow until all leaves are pure.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            max_features (int): The number of features to consider when looking for the best split. If None, all features are considered.
            debug (bool): If True, the logging level will be set to DEBUG, providing more detailed logging information.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        self.random = np.random.RandomState(random_state)
        self.root: Node = None

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def __repr__(self) -> str:
        return f"DecisionTreeClassifier(max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_features={self.max_features})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the decision tree model to the given dataset.

        Parameters:
            X (np.ndarray): The input features array.
            y (np.ndarray): The target values array.
        """
        self._logger.debug("Starting to fit the model.")
        self.root = self._grow_tree(X, y)
        self._logger.debug("Model fitting completed.")

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively grows the decision tree from the given dataset.

        Parameters:
            X (np.ndarray): The input features array.
            y (np.ndarray): The target values array.
            depth (int): The current depth of the tree.

        Returns:
            Node: The root node of the grown tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (
            (depth >= self.max_depth)
            or (n_labels == 1)
            or (n_samples < self.min_samples_split)
        ):
            self._logger.debug(
                f"Reached leaf node. Depth: {depth}, Samples: {n_samples}, Labels: {n_labels}"
            )
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        if self.max_features is not None:
            features_idxs = self.random.choice(
                n_features, self.max_features, replace=False
            )
        else:
            features_idxs = np.arange(n_features)

        self._logger.debug(f"Considering features: {features_idxs}")
        best_feat, best_thresh = self._best_criteria(X, y, features_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            self._logger.debug("No split possible. Creating leaf node.")
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        self._logger.debug(
            f"Splitting at depth {depth}: Feature {best_feat} at threshold {best_thresh}, Left samples: {len(left_idxs)}, Right samples: {len(right_idxs)}"
        )

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(
        self, X: np.ndarray, y: np.ndarray, features_idxs: np.ndarray
    ) -> Tuple[int, float]:
        """
        Finds the best criteria for splitting the dataset.

        Parameters:
            X (np.ndarray): The input features array.
            y (np.ndarray): The target values array.
            features_idxs (np.ndarray): The indices of the features to consider.

        Returns:
            Tuple[int, float]: The index of the best feature and the best threshold for splitting.
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        for idx in features_idxs:
            self._logger.debug(f"Finding best split for feature {idx}")
            feature = X[:, idx]
            thresholds = np.unique(feature)
            for threshold in thresholds:
                gain = self._information_gain(y, feature, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = threshold

        self._logger.debug(
            f"Best split found at feature {split_idx} with threshold {split_thresh} and gain {best_gain}"
        )

        return split_idx, split_thresh

    def _information_gain(
        self, y: np.ndarray, feature: np.ndarray, threshold: float
    ) -> float:
        """
        Calculates the information gain of a potential split.

        Parameters:
            y (np.ndarray): The target values array.
            feature (np.ndarray): The feature values array.
            threshold (float): The threshold for splitting.

        Returns:
            float: The information gain of the split.
        """
        parent_loss = self._entropy(y)

        left_idxs, right_idxs = self._split(feature, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_loss - child_loss

        return ig

    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculates the entropy of a dataset.

        Parameters:
            y (np.ndarray): The target values array.

        Returns:
            float: The entropy of the dataset.
        """
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        entropy = -np.sum(p * np.log2(p))
        return entropy

    def _split(
        self, feature: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits the dataset based on the given feature and threshold.

        Parameters:
            feature (np.ndarray): The feature values array.
            threshold (float): The threshold for splitting.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The indices of the samples in the left and right splits.
        """
        left_idxs = np.argwhere(feature <= threshold).flatten()
        right_idxs = np.argwhere(feature > threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Finds the most common label in the dataset.

        Parameters:
            y (np.ndarray): The target values array.

        Returns:
            int: The most common label.
        """
        if len(y) == 0:
            self._logger.warning("No samples to classify. Returning 0.")
            return None

        common_label = np.bincount(y).argmax()
        self._logger.debug(f"Most common label: {common_label}")
        return common_label

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for the given dataset.

        Parameters:
            X (np.ndarray): The input features array.

        Returns:
            np.ndarray: The predicted class labels.
        """
        self._logger.debug("Starting prediction.")
        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        self._logger.debug("Prediction completed.")
        return predictions

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Union[int, None]:
        """
        Traverses the decision tree to predict the class label for a single sample.

        Parameters:
            x (np.ndarray): The input features for a single sample.
            node (Node): The current node in the decision tree.

        Returns:
            Union[int, None]: The predicted class label, or None if no prediction could be made.
        """
        if node.is_leaf_node():
            self._logger.debug(f"Reached leaf node. Value: {node.value}")
            return node.value

        if x[node.feature] <= node.threshold:
            self._logger.debug(
                f"Traversing left node. Feature: {node.feature}, Threshold: {node.threshold}"
            )
            return self._traverse_tree(x, node.left)

        else:
            self._logger.debug(
                f"Traversing right node. Feature: {node.feature}, Threshold: {node.threshold}"
            )
            return self._traverse_tree(x, node.right)

    def visualize_tree(
        self, feature_names: List[str] = None, class_names: List[str] = None
    ) -> Digraph:
        """
        Visualizes the decision tree.

        Parameters:
            feature_names (List[str]): The names of the features.
            class_names (List[str]): The names of the classes.

        Returns:
            Digraph: A Graphviz Digraph object representing the decision tree.
        """
        dot = Digraph()

        def add_nodes_edges(node: Node, dot: Digraph) -> None:
            """
            Recursively adds nodes and edges to the Graphviz Digraph.

            Parameters:
                node (Node): The current node in the decision tree.
                dot (Digraph): The Graphviz Digraph object.
            """
            if node.is_leaf_node():
                class_name = class_names[node.value] if class_names else str(node.value)
                # Color leaf nodes green
                dot.node(
                    str(id(node)),
                    label=f"Leaf: {class_name}",
                    shape="ellipse",
                    color="lightgreen",
                    style="filled",
                )
            else:
                feature_name = (
                    feature_names[node.feature]
                    if feature_names
                    else f"Feature {node.feature}"
                )
                # Color decision nodes blue
                dot.node(
                    str(id(node)),
                    label=f"{feature_name}\ <= {node.threshold:.2f}",
                    shape="box",
                    color="lightblue",
                    style="filled",
                )
                if node.left:
                    add_nodes_edges(node.left, dot)
                    dot.edge(
                        str(id(node)), str(id(node.left)), label="True", color="black"
                    )
                if node.right:
                    add_nodes_edges(node.right, dot)
                    dot.edge(
                        str(id(node)), str(id(node.right)), label="False", color="red"
                    )

        add_nodes_edges(self.root, dot)
        return dot
