import collections
import numpy as np


class Background:
    def __init__(self, length=20, depth=80):
        self.length = length
        self.depth = depth
        self.__bg_deque = collections.deque(maxlen=length)
        self.__cnt = 0
        self.bg = np.array([]).astype(np.uint8)
        self.active = False  # If there is an initial frame of background

    def __inert(self, frame, depth):
        assert str(type(frame)) == "<class 'numpy.ndarray'>"
        if frame.min() > depth:
            self.__cnt += 1
            self.__bg_deque.append(frame)
            if self.__cnt > 50:
                self.active = True

    def update(self, frame):
        self.__inert(frame, depth=self.depth)
        if self.active:
            self.bg = (np.array(self.__bg_deque).sum(axis=0) // self.length).astype(np.uint8)
        return self.active
