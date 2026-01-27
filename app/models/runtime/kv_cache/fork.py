# app/models/runtime/kv_cache/fork.py
from copy import deepcopy


class KVForkStrategy:
    def fork(self, kv):
        raise NotImplementedError


class DeepCopyFork(KVForkStrategy):
    def fork(self, kv):
        return deepcopy(kv)
