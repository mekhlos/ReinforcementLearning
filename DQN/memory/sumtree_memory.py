from DQN.memory.sumtree import SumTree


class SumTreeMemory:
    """ Memory for prioritised experience replay,
     experiences are sampled with probabilities proportional
     to their priority.

    """

    # determines how much prioritisation is used
    a = 0.6
    prio_max = 0

    def __init__(self, capacity):
        """
        :param capacity: maximum number of experiences that can be fit
         in the memory
        """

        self.capacity = capacity

        # small epsilon value added to the error so that every experience
        # has at least a small chance to be sampled
        self.e = 1.0 / float(capacity)

        # create a SumTree, the data-structure behind this memory
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        """
        :param error: TD error for the given experience
        :return: priority computed based on the error
        """
        return (error + self.e) ** self.a

    def add(self, error, item):
        """ Add new item to the memory with priority
         computed based on error

        :param error: TD error for given experience
        :param item: experience to add to the memory
        """
        p = self._get_priority(error)
        self.tree.add(p, item)

    def add_p(self, item):
        """ Add a new item to the memory with
         maximal priority so that it is sampled at least once
        :param item: experience to add to the memory
        """
        p = max(self.prio_max, self.e)
        self.tree.add(p, item)

    def sample(self, n):
        """ Sample n entries from the memory with probabilities
         proportional to their priority

        :param n: number of entries to sample
        :return: list of samples
        """
        return self.tree.sample(n)

    def update(self, idx, error):
        """ Update priority of a given entry

        :param idx: id of the given experience
        :param error: new TD-error
        """
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def total(self):
        """
        :return: sum of priorities of the entries in the memory
        """
        return self.tree.total()

    def get_size(self):
        """
        :return: current number of entries in the memory
        """
        return self.tree.n_entries

    def update_max(self, idx):
        """ compute maximum priority in the memory

        :param idx: id-s of most recently updated experiences
        """
        self.prio_max = max(self.tree.get_priorities(idx).max(), self.prio_max)

    def is_full(self):
        """
        :return: True if the number of entries in the memory is
                    equal to the capacity of the memory
        """
        return self.get_size() == self.capacity
