from __future__ import annotations

import math

import numpy as np
import torch
from numba import njit

from .kernels import ensure_float32_contiguous


def make_attention_inputs(
    seq_len: int,
    head_dim: int,
    value_dim: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value_dim = head_dim if value_dim is None else value_dim
    rng = np.random.default_rng(seed)
    q = ensure_float32_contiguous(rng.standard_normal((seq_len, head_dim), dtype=np.float32))
    k = ensure_float32_contiguous(rng.standard_normal((seq_len, head_dim), dtype=np.float32))
    v = ensure_float32_contiguous(rng.standard_normal((seq_len, value_dim), dtype=np.float32))
    return q, k, v


@njit(cache=True)
def materialized_attention_cpu(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    seq_len, head_dim = q.shape
    _, value_dim = v.shape
    scale = 1.0 / math.sqrt(head_dim)

    scores = np.empty((seq_len, seq_len), dtype=np.float32)
    output = np.zeros((seq_len, value_dim), dtype=np.float32)

    for row in range(seq_len):
        row_max = -1.0e30
        for col in range(seq_len):
            acc = 0.0
            for kk in range(head_dim):
                acc += q[row, kk] * k[col, kk]
            score = acc * scale
            scores[row, col] = score
            if score > row_max:
                row_max = score

        denom = 0.0
        for col in range(seq_len):
            weight = math.exp(scores[row, col] - row_max)
            scores[row, col] = weight
            denom += weight

        inv_denom = 1.0 / denom
        for col in range(seq_len):
            weight = scores[row, col] * inv_denom
            for vd in range(value_dim):
                output[row, vd] += weight * v[col, vd]

    return output


# Backward-compatible alias for the first draft of the benchmark.
naive_attention_cpu = materialized_attention_cpu


@njit(cache=True)
def streaming_attention_cpu(q: np.ndarray, k: np.ndarray, v: np.ndarray, block_size: int = 64) -> np.ndarray:
    seq_len, head_dim = q.shape
    _, value_dim = v.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = np.zeros((seq_len, value_dim), dtype=np.float32)

    for row in range(seq_len):
        running_max = -1.0e30
        running_sum = 0.0
        running_out = np.zeros(value_dim, dtype=np.float32)

        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            local_max = -1.0e30
            scores_block = np.empty(end - start, dtype=np.float32)

            for idx in range(end - start):
                col = start + idx
                acc = 0.0
                for kk in range(head_dim):
                    acc += q[row, kk] * k[col, kk]
                score = acc * scale
                scores_block[idx] = score
                if score > local_max:
                    local_max = score

            new_max = running_max if running_max > local_max else local_max
            old_scale = 0.0 if running_sum == 0.0 else math.exp(running_max - new_max)
            block_sum = 0.0
            block_out = np.zeros(value_dim, dtype=np.float32)

            for idx in range(end - start):
                weight = math.exp(scores_block[idx] - new_max)
                block_sum += weight
                col = start + idx
                for vd in range(value_dim):
                    block_out[vd] += weight * v[col, vd]

            for vd in range(value_dim):
                running_out[vd] = running_out[vd] * old_scale + block_out[vd]

            running_sum = running_sum * old_scale + block_sum
            running_max = new_max

        inv_sum = 1.0 / running_sum
        for vd in range(value_dim):
            output[row, vd] = running_out[vd] * inv_sum

    return output


def torch_attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_t = torch.from_numpy(q)
    k_t = torch.from_numpy(k)
    v_t = torch.from_numpy(v)
    scale = q.shape[1] ** -0.5
    scores = torch.matmul(q_t, k_t.T) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_t).numpy()
