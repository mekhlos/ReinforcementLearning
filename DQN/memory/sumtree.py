""" Necessary to build a replay memory for proportional prioritised
 experience replay.

"""
# Inspired by: https://github.com/jaara/AI-blog/blob/master/SumTree.py

import numpy as np


class SumTree:
    """ Tree data-structure where each parent's value is the sum of the
     values of its children

    """

    def __init__(self, capacity):
        """
        :param capacity: number of entries that can be stores in the leaves
        """

        self.capacity = capacity

        # Each parent has two children therefore we have 2 * n - 1 nodes for
        # n leaves
        self.tree = np.zeros(2 * capacity - 1)

        # to store leaves
        self.data = np.zeros(capacity, dtype=object)

        self.write = 0
        self.n_entries = 0

        self.tree_len = len(self.tree)

    def _propagate(self, idx, change):
        """ Propagate change upwards in the tree
         (triggered when a leaf node is changed)

        :param idx: index of current node
        :param change: amount of change to propagate
        """

        # compute parent's index from index of child
        parent_id = (idx - 1) // 2

        self.tree[parent_id] += change

        if parent_id != 0:
            self._propagate(parent_id, change)

    def _retrieve(self, idx, s):
        """ Retrieves a leaf node with probability proportional to its value

        :param idx: index of current node
        :param s: random number uniformly chosen between 0 and
                    sum of all leaf node values
        :return: a leaf node
        """
        left_child_id = 2 * idx + 1
        right_child_id = left_child_id + 1

        if left_child_id >= self.tree_len:
            return idx

        if s <= self.tree[left_child_id]:
            return self._retrieve(left_child_id, s)
        else:
            return self._retrieve(right_child_id, s - self.tree[left_child_id])

    def total(self):
        """
        :return: sum of all leaf node values i.e. value of root node
        """
        return self.tree[0]

    def add(self, p, data):
        """ Add a new entry with priority p to the SumTree

        :param p: priority of new entry
        :param data: new entry
        """

        # compute new id
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.capacity, self.n_entries + 1)

    def update(self, idx, p):
        """ Update entry with index idx in the tree

        :param idx: index of the entry
        :param p: new priority
        """

        # difference between the old and new values of the given entry
        change = p - self.tree[idx]

        self.tree[idx] = p

        # propagate update up in the tree
        self._propagate(idx, change)

    def get(self, s):
        """ Return the value of a leaf node with probability proportional
         to its priority

        :param s: random number generated uniformly between 0 and sum of all
                leaf node values
        :return: an id, the priority and the entry itslf
        """
        idx = self._retrieve(0, s)
        data_id = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_id]

    # https://gist.github.com/avalcarce/d6a16387a364fd366e5de49446e2dd6b
    def sample(self, n):
        """ Sample n entries from the tree

        :param n: number of items to sample
        :return: a list of ids, a list of priorities and a list of entries
        """

        # initialise lists containing samples
        batch_id = [None] * n
        batch_priorities = [None] * n
        batch = [None] * n

        segment = self.total() / n

        a = [segment * i for i in range(n)]
        b = [segment * (i + 1) for i in range(n)]
        s = np.random.uniform(a, b)

        for i in range(n):
            (batch_id[i], batch_priorities[i], batch[i]) = self.get(s[i])

        return batch_id, batch_priorities, batch

    def get_priorities(self, idx):
        return self.tree[idx]
