"""Exact-input, bounded single-flight query embeddings."""
from collections import OrderedDict
from concurrent.futures import Future
from threading import RLock
from typing import Callable

import numpy as np


class QueryEmbeddingCache:
    def __init__(self, capacity: int = 64):
        self.capacity = max(0, capacity)
        self.lock = RLock()
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()
        self.pending: dict[str, Future] = {}
        self._generation = 0

    def clear(self):
        with self.lock:
            self.values.clear()
            self.pending.clear()
            self._generation += 1

    def get(self, text: str, embed: Callable[[str], np.ndarray]) -> np.ndarray:
        with self.lock:
            if self.capacity > 0 and text in self.values:
                self.values.move_to_end(text)
                return self.values[text]
            pending = self.pending.get(text)
            owner = pending is None
            generation = self._generation
            if owner:
                pending = Future()
                self.pending[text] = pending
        assert pending is not None
        if not owner:
            return pending.result()
        try:
            value = np.array(embed(text), copy=True)
            if value.ndim != 1 or not value.size or not np.all(np.isfinite(value)):
                raise ValueError("Query embedding must be a nonempty finite vector")
            value.setflags(write=False)
            with self.lock:
                if generation == self._generation and self.capacity > 0 and text:
                    self.values[text] = value
                    while len(self.values) > self.capacity:
                        self.values.popitem(last=False)
            pending.set_result(value)
            return value
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self.lock:
                if self.pending.get(text) is pending:
                    self.pending.pop(text, None)
