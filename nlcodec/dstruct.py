#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-10-25

from typing import List, Dict, Union, Optional, TypeVar, Generic, Any
import heapq


class LnNode:  # doubly linked list node data structure; used for learning BPE
    __slots__ = 'val', 'left', 'right', 'freq', 'data'

    def __init__(self, val: int, left: Optional['LnNode'] = None,
                 right: Optional['LnNode'] = None, freq: int = 1, data: Any = None):
        self.val = val
        self.left = left
        self.right = right
        self.freq = freq
        self.data = data

    def __eq__(self, other):
        # caution: these calls are recursive on left and right; cycles would cause infinite loop
        return (other.val == self.val and other.freq == self.freq and
                other.left == self.left and other.right == self.right)

    def __hash__(self):
        return id(self)  # quick and dirty hash; not sure how this mess if we use multiprocessing

    def delete(self, unlink=True):
        """
        deletes this node from the list
        :return:
        """
        x, y = self.left, self.right
        if x:  # right links : x → self → y  => x → y
            x.right = y
        if y:  # left links  : x ← self ← y  => x ← y
            y.left = x
        if unlink:
            self.left = self.right = None

    @property
    def is_unlinked(self):
        return self.right is None and self.left is None

    @classmethod
    def from_seq(cls, string: Union[str, List[int]], freq=1, data=None) -> List['LnNode']:
        """
        makes a doubly linked list from string
        :param string: input string (of integers recommended)
        :param freq: frequency of string in corpus (for scaling
        :return: List of Nodes, doubly linked to lefts and rights
        """
        nodes = [cls(ch, freq=freq, data=data) for ch in string]
        for i, n in enumerate(nodes):
            if i > 0:
                n.left = nodes[i - 1]
            if i + 1 < len(nodes):
                n.right = nodes[i + 1]
        return nodes

    def __repr__(self):
        lefts, rights = [], []

        cur = self.left
        while cur:
            lefts.append(str(cur.val))
            cur = cur.left
        cur = self.right
        while cur:
            rights.append(str(cur.val))
            cur = cur.right
        return ' '.join(reversed(lefts)) + f' *{self.val}* ' + ' '.join(rights)


T = TypeVar('T')  # T for index; Dont use I, since 'I' agree with pep008
D = TypeVar('D')  # D for data


class TrNode(Generic[T, D]):  # Trie Node or Tree Node

    __slots__ =  'idx', 'name', 'data', 'parent', 'kids'

    def __init__(self, idx: T, name: Optional[str] = None, data: Optional[D] = None,
                 parent: Optional['TrNode'] = None, kids: Dict[T, 'TrNode'] = None):
        self.idx = idx
        self.name = name
        self.data = data
        self.parent = parent
        self.kids = dict() if kids is None else kids

    def get_node(self, idxs, create_missing: bool = True) -> 'TrNode[T, D]':
        if not idxs:
            return self
        if create_missing and idxs[0] not in self.kids:  # make one
            self.kids[idxs[0]] = TrNode(idx=idxs[0], parent=self)
        return self.kids[idxs[0]].get_node(idxs=idxs[1:], create_missing=create_missing)

    @property
    def n_kids(self):
        """Number of immediate children"""
        return len(self.kids)

    @property
    def has_data(self):
        return self.data is not None

    @property
    def size(self):
        """Number of nodes in the tree at this node. Counts self and all the kids"""
        return 1 + sum(k.size for k in self.kids.values())

    @property
    def data_node_count(self):
        return (1 if self.has_data else 0) + sum(k.data_node_count for k in self.kids.values())


class MaxHeap:
    "Offers Max Heap which is a wrapper to fast min heap implementation from heapq "

    def __init__(self, items: Dict[Any, int]):
        self.min_heap = [(-val, node) for node, val in items.items()]
        heapq.heapify(self.min_heap)

    def push(self, node, val):
        """
        Adds a node to heap while maintaining heap property
        :param node: node data
        :param val: priority value
        :return:
        """
        heapq.heappush(self.min_heap, (-val, node))

    def pop(self):
        """returns the max item (task, priority)"""
        val, task = heapq.heappop(self.min_heap)
        return task, -val

    def peek(self):
        val, task = self.min_heap[0]
        return task, -val

    def __len__(self):
        return len(self.min_heap)
