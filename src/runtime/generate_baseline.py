"""
Baseline generation (no KV cache).

Generates max_new_tokens tokens by recomputing the full forward pass on the
entire sequence each step (slow baseline).

Measures:
- prefill time (ms): processing the prompt once
- decode time (ms): each subsequent autoregressive step (recomputes prefix)
"""

from tqdm import tqdm
import torch
import time


def generate_baseline(model, input_ids, max_new_tokens, use_tqdm=True, leave=True):
    """
    Returns:
      (output_ids, prefill_ms, decode_ms_total, decode_ms_per_step_list)
    """
    assert input_ids.dtype == torch.long, "input_ids must be torch.long"
    assert input_ids.device == next(model.parameters()).device, "model/input_ids device mismatch"

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

        logits = model(input_ids)

        if use_cuda_timing:
            end_prefill.record()
            torch.cuda.synchronize()
            prefill_time_ms = start_prefill.elapsed_time(end_prefill)
        else:
            prefill_time_ms = (time.perf_counter() - t0) * 1000.0

        # first generated token comes from prefill logits
        next_id = torch.argmax(logits[:, -1, :], dim=-1).to(torch.long).unsqueeze(1)  # (B, 1)

        # if already at max length, stop
        if input_ids.shape[1] >= model.max_seq_len:
            return input_ids, prefill_time_ms, decode_time_total_ms, decode_times_ms

        input_ids = torch.cat([input_ids, next_id], dim=1)
        generated = 1


        # Decode loop
        steps = min(max_new_tokens - 1, model.max_seq_len - input_ids.shape[1])
        start_wall = time.perf_counter()

        iterator = range(steps)
        if use_tqdm:
            iterator = tqdm(iterator, desc="Baseline Decode", leave=leave)

        for _ in iterator:
            if use_cuda_timing:
                start_decode.record()
            else:
                t1 = time.perf_counter()

            logits = model(input_ids)

            if use_cuda_timing:
                end_decode.record()
                torch.cuda.synchronize()
                step_ms = start_decode.elapsed_time(end_decode)
            else:
                step_ms = (time.perf_counter() - t1) * 1000.0

            decode_time_total_ms += step_ms
            decode_times_ms.append(step_ms)

            next_id = torch.argmax(logits[:, -1, :], dim=-1).to(torch.long).unsqueeze(1)
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
