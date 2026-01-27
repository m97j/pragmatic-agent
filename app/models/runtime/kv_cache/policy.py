# app/models/runtime/kv_cache/policy.py
class KVCachePolicy:
    def should_fork(self, step: int) -> bool:
        return True

    def allow_reuse(self) -> bool:
        return True


class SpeculativeKVPolicy(KVCachePolicy):
    def __init__(self, max_accept: int):
        self.max_accept = max_accept

    def should_fork(self, step: int) -> bool:
        return step < self.max_accept
