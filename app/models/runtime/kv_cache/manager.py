# app/models/runtime/kv_cache/manager.py
import uuid

from app.models.runtime.kv_cache.eviction import NoEviction
from app.models.runtime.kv_cache.fork import DeepCopyFork


class KVCacheManager:
    def __init__(
        self,
        fork_strategy=None,
        eviction_policy=None,
    ):
        self.fork_strategy = fork_strategy or DeepCopyFork()
        self.eviction_policy = eviction_policy or NoEviction()
        self.cache_pool = {}

    def register(self, kv):
        key = str(uuid.uuid4())
        self.cache_pool[key] = kv
        self.eviction_policy.evict(self.cache_pool)
        return key

    def get(self, key):
        return self.cache_pool[key]

    def fork(self, kv):
        forked = self.fork_strategy.fork(kv)
        return self.register(forked)

    def rollback(self, kv, step: int):
        """
        Rollback KV to `step`
        step == number of accepted tokens
        """
        rolled = []
        for layer_k, layer_v in kv:
            rolled.append((
                layer_k[:, :, :step, :],
                layer_v[:, :, :step, :]
            ))
        return rolled

    def drop(self, key):
        if key in self.cache_pool:
            del self.cache_pool[key]
