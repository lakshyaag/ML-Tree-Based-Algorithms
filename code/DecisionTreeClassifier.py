import logging

import numpy as np
from graphviz import Digraph

from Node import Node


class DecisionTreeClassifier:
    def __init__(
        self, max_depth=None, min_samples_split=2, max_features=None, debug=False
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        self.random = np.random.RandomState(42)
        self.root = None

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO)

        if debug:
            self._logger.setLevel(logging.DEBUG)

    def __repr__(self):
        return f"DecisionTreeClassifier(max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_features={self.max_features})"

    def fit(self, X, y):
        self._logger.debug("Starting to fit the model.")
        self.root = self._grow_tree(X, y)
        self._logger.debug("Model fitting completed.")

    def _grow_tree(self, X, y, depth=0):
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

    def _best_criteria(self, X, y, features_idxs):
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

    def _information_gain(self, y, feature, threshold):
        parent_loss = self._entropy(y)
        # self._logger.debug(f"Parent loss: {parent_loss}")

        left_idxs, right_idxs = self._split(feature, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        # self._logger.debug(f"Left entropy: {e_l}, Right entropy: {e_r}")
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_loss - child_loss

        # self._logger.debug(f"Information gain at threshold {threshold}: {ig}")

        return ig

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        entropy = -np.sum(p * np.log2(p))
        return entropy

    def _split(self, feature, threshold):
        left_idxs = np.argwhere(feature <= threshold).flatten()
        right_idxs = np.argwhere(feature > threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        if len(y) == 0:
            self._logger.warning("No samples to classify. Returning 0.")
            return None

        common_label = np.bincount(y).argmax()
        self._logger.debug(f"Most common label: {common_label}")
        return common_label

    def predict(self, X):
        self._logger.debug("Starting prediction.")
        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        self._logger.debug("Prediction completed.")
        return predictions

    def visualize_tree(self, feature_names=None, class_names=None):
        dot = Digraph()

        def add_nodes_edges(node, dot):
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
                    label=f"{feature_name}\nThreshold {node.threshold:.2f}",
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

    def _traverse_tree(self, x, node: Node):
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
