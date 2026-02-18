"""
This file contains the implementation of the KV cache.

The primary purpose of such a cache is for each layer to store the Key and Value tensors for past tokens so we don't 
have to recompute them during decode.

KV cache is part of the runtime state (not a model parameter).
"""

import torch

class KVCache:
    def __init__(self, config, batch_size):
        # I initially thought of implementing the k and v cache using a hashmap
        # however, Python objects are slow and IRL inference engines implement the cache with preallocated contiguous GPU memory
        # each layer has its own K and V history; in physical memory, they are slices of this one big tensor
        # we initialize with empty buffers of zeros
        self.k_cache = torch.zeros(config.num_layers, batch_size, config.num_heads, config.max_seq_len, config.head_dim, device=config.device, dtype=config.dtype)
        self.v_cache = torch.zeros(config.num_layers, batch_size, config.num_heads, config.max_seq_len, config.head_dim, device=config.device, dtype=config.dtype)

        # keeps track of how many tokens are currently in cache
        self.cur_len = 0

        # get params from config
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers


    # during prefill we compute the K/V for the prompt all at once in parallel
    # then we store them all at once with set
    # bulk write (prefill)
    def set(self, layer_idx, k_full, v_full, seq_len):
        # make sure our caches have same shape
        assert k_full.shape == v_full.shape, "K and V must have identical shapes"
        assert self.k_cache.shape == self.v_cache.shape

        B, Nh, T, Hd = k_full.shape

        # structural checks
        assert layer_idx >= 0 and layer_idx < self.k_cache.shape[0], "Invalid layer index"
        assert T == seq_len, "seq_len must match K/V tensor length"
        assert T <= self.k_cache.shape[3], "seq_len exceeds max_seq_len"

        # device and dtype check
        assert k_full.device == self.k_cache.device, "Device mismatch for k cache"
        assert v_full.device == self.v_cache.device, "Device mismatch for v cache"
        assert k_full.dtype == self.k_cache.dtype, "Dtype mismatch for k cache"
        assert v_full.dtype == self.v_cache.dtype, "Dtype mismatch for v cache"

        self.k_cache[layer_idx, :, :, :seq_len, :] = k_full
        self.v_cache[layer_idx, :, :, :seq_len, :] = v_full
        self.cur_len = seq_len


    # read the KV cache up to the current length
    def get_kv(self, layer_idx):
        assert layer_idx >= 0 and layer_idx < self.k_cache.shape[0], "Invalid layer index"
        max_seq_len = self.k_cache.shape[3]
        assert 0 <= self.cur_len <= max_seq_len, "cur_len out of bounds"

        k_cached = self.k_cache[layer_idx, :, :, :self.cur_len, :]
        v_cached = self.v_cache[layer_idx, :, :, :self.cur_len, :]
        return k_cached, v_cached

    # append new k and new v vector to the KV cache
    # write 1 position
    def append(self, layer_idx, k_new, v_new, pos):
        B, Nh, T, Hd = k_new.shape
        assert k_new.shape == v_new.shape, "K and V must match"
        assert pos >= 0 and pos < self.k_cache.shape[3], "Position out of bounds"
        assert T == 1, "Decode only processes 1 new token at a time"

        # device and dtype check
        assert k_new.device == self.k_cache.device, "Device mismatch for k cache"
        assert v_new.device == self.v_cache.device, "Device mismatch for v cache"
        assert k_new.dtype == self.k_cache.dtype, "Dtype mismatch for k cache"
        assert v_new.dtype == self.v_cache.dtype, "Dtype mismatch for v cache"

        self.k_cache[layer_idx, :, :, pos:pos+1, :] = k_new
        self.v_cache[layer_idx, :, :, pos:pos+1, :] = v_new

        # I decided to advance the current length in a separate way
        # if we do this then we might be prone to advancing the length for each layer in our model instead of each token appended
        #self.cur_len += 1

    # reset the KV cache for multiple generations
    def reset(self):
        self.cur_len = 0


