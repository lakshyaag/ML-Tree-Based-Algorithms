class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

    def __repr__(self):
        if self.is_leaf_node():
            return f"Node(value={self.value})"
        return f"Node(feature={self.feature}, threshold={self.threshold}, left={self.left}, right={self.right})"
