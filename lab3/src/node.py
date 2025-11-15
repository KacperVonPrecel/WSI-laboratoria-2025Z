from typing import Any, Callable


class Node():
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.children = {}
        self.label = label
