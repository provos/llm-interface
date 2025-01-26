import logging
from collections import defaultdict
from threading import Lock
from typing import Dict


# Mock Cache
class MockCache:
    def __init__(self, dont_store=False):
        self.store = {}
        self._dont_store = dont_store
        self.set_stats: Dict[str, int] = defaultdict(int)
        self.get_stats: Dict[str, int] = defaultdict(int)
        self.lock = Lock()

    def get(self, key, default=None):
        logging.debug("Getting key: %s", key)
        with self.lock:
            self.get_stats[key] += 1
            return self.store.get(key, default)

    def set(self, key, value):
        if self._dont_store:
            return
        logging.debug("Setting key: %s", key)
        with self.lock:
            self.store[key] = value
            self.set_stats[key] += 1

    def clear_stats(self):
        self.set_stats.clear()
        self.get_stats.clear()
