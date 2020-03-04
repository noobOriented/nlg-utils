from itertools import chain


class SuffixTree:

    def __init__(self):
        self.root = TrieNode()

    def add(self, word):
        if isinstance(word, str):
            word = memoryview(word.encode())

        active_point = ActivePoint(self.root, word)
        new_leaves = []
        dummy_node = TrieNode()
        for end, c in enumerate(word):
            active_point.end = end  # extension A
            prev = dummy_node
            for _ in range(end + 1 - len(new_leaves)):
                if active_point.match(c):
                    # extension C
                    active_point.length += 1
                    prev.suffix = active_point.node
                    break
                # extension B
                new_leaf = TrieNode(word[end:])
                new_leaves.append(new_leaf)
                internal_node = active_point.insert_leaf(new_leaf)
                prev.suffix = internal_node
                prev = internal_node

                active_point.goto_suffix()

    def longest_substring_length(self, word):
        if isinstance(word, str):
            word = word.encode()

        max_depth = 0
        active_point = ActivePoint(self.root, word)
        for end, c in enumerate(word):
            active_point.end = end
            for _ in range(active_point.depth + 1):
                if active_point.match(c):
                    active_point.length += 1
                    break

                active_point.goto_suffix()

            max_depth = max(active_point.depth, max_depth)

        return max_depth

    def all_occurrences(self, word):
        if isinstance(word, str):
            word = word.encode()

        active_point = ActivePoint(self.root, word)
        for end, c in enumerate(word):
            active_point.end = end
            if active_point.match(c):
                active_point.length += 1
            else:
                return []

        active_point.end = len(word)
        return [-leaf.depth for leaf in active_point.edge.get_leaves()]


class TrieNode:

    def __init__(self, data=''):
        self.data = data

        self.children = {}
        self.parent = None
        self._depth = None
        self.suffix = None

    def add_child(self, child):
        self.children[child.data[0]] = child
        child.parent = self

    @property
    def depth(self):
        if not self.parent:
            return 0  # root
        if not self._depth:
            self._depth = self.parent.depth + len(self.data)
        return self._depth

    def get_leaves(self):
        if not self.children:
            return [self]
        return list(chain.from_iterable(
            child.get_leaves() for child in self.children.values()
        ))


class ActivePoint:

    def __init__(self, root, word):
        self.word = word
        self.node = root
        self.end = None
        self.length = 0

    @property
    def edge(self):
        if self.length == 0:
            return None
        key = self.word[self.end - self.length]
        return self.node.children[key]

    @property
    def depth(self):
        return self.node.depth + self.length

    def match(self, c):
        while True:
            edge = self.edge
            if not edge or self.length < len(edge.data):
                break
            self.length -= len(edge.data)
            self.node = edge

        if self.length == 0:
            return c in self.node.children
        return c == self.edge.data[self.length]

    def insert_leaf(self, leaf):
        if self.length:
            edge = self.edge
            internal_node = TrieNode(edge.data[:self.length])
            edge.data = edge.data[self.length:]
            internal_node.add_child(edge)
            self.node.add_child(internal_node)
        else:
            internal_node = self.node

        internal_node.add_child(leaf)
        return internal_node

    def goto_suffix(self):
        if self.node.suffix:  # is not root
            self.node = self.node.suffix
        elif self.length:
            self.length -= 1
