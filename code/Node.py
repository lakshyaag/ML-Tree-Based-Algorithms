from __future__ import annotations


class Node:
    """
    A class used to represent a node in a decision tree.

    Attributes:
        feature (int): The index of the feature used for splitting the data.
        threshold (float): The threshold value for the split at this node.
        left (Node): The left child node resulting from the split.
        right (Node): The right child node resulting from the split.
        value (int, optional): The class value assigned to this node if it is a leaf node.

    """

    def __init__(
        self,
        feature: int = None,
        threshold: int = None,
        left: Node = None,
        right: Node = None,
        *,
        value: int = None,
    ) -> None:
        """
        Initializes the Node with the provided feature, threshold, children, and value.

        Parameters:
            feature (int): The index of the feature used for splitting the data.
            threshold (float): The threshold value for the split at this node.
            left (Node): The left child node resulting from the split.
            right (Node): The right child node resulting from the split.
            value (int, optional): The class value assigned to this node if it is a leaf node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        """
        Determines if the node is a leaf node.

        Returns:
            bool: True if the node is a leaf node, False otherwise.
        """
        return self.value is not None

    def __repr__(self) -> str:
        """
        Represents the Node as a string.

        Returns:
            str: A string representation of the Node.
        """
        if self.is_leaf_node():
            return f"Node(value={self.value})"
        return f"Node(feature={self.feature}, threshold={self.threshold}, left={self.left}, right={self.right})"
