"""
This script generates max_new_tokens tokens given a model and a starting prompt.

input_ids is the prompt.

Measures:
- prefill time (ms): processing the prompt (one full forward)
- decode time (ms): subsequent per-token decode_step calls

Uses KVCache to avoid recomputing K/V for past tokens.
"""

from tqdm import tqdm
import torch
import time

from src.runtime.inference import prefill, decode_step
from src.runtime.kv_cache import KVCache


def generate_kv(model, input_ids, max_new_tokens, use_tqdm=True, leave=True):
    """
    Returns:
      (output_ids, prefill_ms, decode_ms_total, decode_ms_per_step_list)
    """
    assert input_ids.dtype == torch.long, "input_ids must be torch.long"
    assert input_ids.device == next(model.parameters()).device, "model/input_ids device mismatch"

    B, T = input_ids.shape

    # create KV cache for this request
    kv_cache = KVCache(config=model.config, batch_size=B)

    # timing
    use_cuda_timing = torch.cuda.is_available() and input_ids.device.type == "cuda"

    if use_cuda_timing:
        start_prefill = torch.cuda.Event(enable_timing=True)
        end_prefill = torch.cuda.Event(enable_timing=True)
        start_decode = torch.cuda.Event(enable_timing=True)
        end_decode = torch.cuda.Event(enable_timing=True)
    else:
        start_prefill = end_prefill = start_decode = end_decode = None

    prefill_time_ms = 0.0
    decode_time_total_ms = 0.0
    decode_times_ms = []

    model.eval()
    with torch.no_grad():
        # Prefill (prompt)
        if use_cuda_timing:
            start_prefill.record()
        else:
            t0 = time.perf_counter()

        logits = prefill(model, prompt_ids=input_ids, kv_cache=kv_cache)

        if use_cuda_timing:
            end_prefill.record()
            torch.cuda.synchronize()
            prefill_time_ms = start_prefill.elapsed_time(end_prefill)
        else:
            prefill_time_ms = (time.perf_counter() - t0) * 1000.0

        # prefill should have filled cache to length T
        assert kv_cache.cur_len == T, f"kv_cache.cur_len={kv_cache.cur_len} != prompt_len={T}"

        # first generated token comes from prefill logits
        next_id = torch.argmax(logits[:, -1, :], dim=-1).to(torch.long).unsqueeze(1)  # (B, 1)

        # if we're already at max length, stop (no room to append)
        if input_ids.shape[1] >= model.max_seq_len:
            return input_ids, prefill_time_ms, decode_time_total_ms, decode_times_ms

        input_ids = torch.cat([input_ids, next_id], dim=1)
        generated = 1  # number of generated tokens so far (not counting prompt)

        # decode with KV
        steps = min(max_new_tokens - 1, model.max_seq_len - input_ids.shape[1])
        start_wall = time.perf_counter()

        iterator = range(steps)
        if use_tqdm:
            iterator = tqdm(iterator, desc="KV Decode", leave=leave)

        for _ in iterator:
            if use_cuda_timing:
                start_decode.record()
            else:
                t1 = time.perf_counter()

            # decode_step consumes the last generated token (next_id) and returns logits for the next token
            last_logits = decode_step(model, next_id, kv_cache=kv_cache)  # (B, vocab)

            if use_cuda_timing:
                end_decode.record()
                torch.cuda.synchronize()
                step_ms = start_decode.elapsed_time(end_decode)
            else:
                step_ms = (time.perf_counter() - t1) * 1000.0

            decode_time_total_ms += step_ms
            decode_times_ms.append(step_ms)

            next_id = torch.argmax(last_logits, dim=-1).to(torch.long).unsqueeze(1)  # (B, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            generated += 1

            # live throughput metrics 
            if use_tqdm:
                elapsed_wall = time.perf_counter() - start_wall
                if elapsed_wall > 0:
                    tok_per_sec = generated / elapsed_wall
                    ms_per_tok = (elapsed_wall / generated) * 1000.0
                    iterator.set_postfix({
                        "tok/s": f"{tok_per_sec:.1f}",
                        "ms/tok": f"{ms_per_tok:.2f}",
                        "seq_len": input_ids.shape[1],
                    })

    return input_ids, prefill_time_ms, decode_time_total_ms, decode_times_ms
