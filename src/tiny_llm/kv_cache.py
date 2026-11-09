from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import mlx.core as mx

if TYPE_CHECKING:
    from .paged_kv_cache import PagedKvMetadata


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            The updated keys, updated values, sequence length, and mask. On
            On Week 2 Day 1, the mask is passed through unchanged. Week 3 Day 1
            uses the sequence length and mask to construct a dense batch.
        """

    def release(self):
        pass

    def materialize(self):
        """Evaluate owned K/V storage without changing its logical layout."""
        pass

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        raise NotImplementedError

    def rewind(self, n: int):
        raise NotImplementedError


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int | None = None):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        pass

    def update_and_fetch_paged(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self, capacity: int | None = None):
        if capacity is not None and (
            not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0
        ):
            raise ValueError("capacity must be a non-negative integer or None")
        self.key_values = None
        self.offset = 0
        self.capacity = capacity
        self.logical_copy_bytes = 0
        self.physical_growth_copy_bytes = 0
        self.slice_write_bytes = 0
        self.growth_copy_bytes = 0

    @property
    def uses_capacity(self) -> bool:
        return self.capacity is not None

    def _logical_key_values(self) -> tuple[mx.array, mx.array]:
        """Return only initialized tokens, never the unused physical tail."""
        assert self.key_values is not None
        # if self.uses_capacity:
        key, value = self.key_values
        # B, H, S, D (最后那个可以省略)
        return key[:, :, : self.offset, :], value[:, :, : self.offset]
        # else:
        #     # 实际上两者等价
        #     return self.key_values

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        B, H, S, D = key.shape

        if self.uses_capacity:
            end = self.offset + S
            if end > self.capacity:
                raise ValueError("out of capacity")
            if self.key_values is None:
                self.key_values = (
                    mx.zeros((B, H, self.capacity, D), dtype=key.dtype),
                    mx.zeros((B, H, self.capacity, D), dtype=value.dtype),
                )
            prev_k, prev_v = self.key_values
            # 两种写法都 ok
            prev_k = mx.slice_update(
                prev_k,
                key,
                start_indices=mx.array([self.offset], dtype=mx.int32),
                axes=[2],
            )
            prev_v[:, :, self.offset : end, :] = value
            self.key_values = (prev_k, prev_v)
            self.offset = end
        else:
            if self.key_values is None:
                self.key_values = (key, value)
            else:
                prev_k, prev_v = self.key_values
                self.key_values = (
                    mx.concatenate([prev_k, key], axis=2),
                    mx.concatenate([prev_v, value], axis=2),
                )
            self.offset += S

        cached_k, cached_v = self._logical_key_values()
        return cached_k, cached_v, self.offset, mask

    def materialize(self):
        pass

    def reset(self):
        """Reset logical length while retaining bounded physical storage."""
        pass

    def rewind(self, n: int):
        pass
