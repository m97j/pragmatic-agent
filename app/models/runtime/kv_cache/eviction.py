# app/models/runtime/kv_cache/eviction.py
class EvictionPolicy:
    def evict(self, cache_pool: dict):
        raise NotImplementedError


class NoEviction(EvictionPolicy):
    def evict(self, cache_pool: dict):
        return


class LRUEviction(EvictionPolicy):
    def __init__(self, max_entries: int):
        self.max_entries = max_entries

    def evict(self, cache_pool: dict):
        if len(cache_pool) <= self.max_entries:
            return
        # remove the oldest key
        oldest = next(iter(cache_pool.keys()))
        del cache_pool[oldest]
