"""
Script that runs baseline prefill and decode.
"""

import torch
from src.model.config import Config
from src.model.transformer import Transformer
from src.runtime.generate_baseline import generate_baseline
from src.runtime.generate_kv import generate_kv


def make_random_prompt(model, batch_size: int, seq_len: int):
    # get device from model
    device = next(model.parameters()).device

    # generate random token IDs in [0, vocab_size)
    prompt = torch.randint(
        low=0,
        high=model.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    return prompt


def main():
    # build a config and a model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config(
        vocab_size=1024,
        hidden_size=1024,
        num_layers=12,
        num_heads=8,
        max_seq_len=1024,
        max_batch_size=1,
        dtype=torch.float32,
        device=device
    )

    model = Transformer(config=cfg)

    # create synthetic prompt ids
    batch_size = 1
    prompt_len = 10

    prompt = make_random_prompt(model, batch_size, prompt_len)

    # set max new tokens
    max_new_tokens = 1000

    # warmup block
    # without warmup we measure starup overhead like CUDA context init etc.
    warmup_runs = 2
    warmup_tokens = min(16, max_new_tokens)

    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            # warmup baseline generation
            _ = generate_baseline(model=model, input_ids=prompt, max_new_tokens=warmup_tokens, use_tqdm=False)

            # warmup KV generation
            _ = generate_kv(model=model, input_ids=prompt, max_new_tokens=warmup_tokens, use_tqdm=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # create CUDA events with timing enabled
    start_base = torch.cuda.Event(enable_timing=True)
    end_base = torch.cuda.Event(enable_timing=True)

    # create CUDA events with timing enabled
    start_kv = torch.cuda.Event(enable_timing=True)
    end_kv= torch.cuda.Event(enable_timing=True)

    start_base.record()

    # call the generate function
    returned_tokens_base, prefill_time_base, decode_time_total_base, decode_times_base = generate_baseline(model=model, input_ids=prompt, max_new_tokens=max_new_tokens)

    end_base.record()

    # wait for all queued GPU work to finish before querying time on CPU
    torch.cuda.synchronize()

    start_kv.record()

    # call the generate function
    returned_tokens_kv, prefill_time_kv, decode_time_total_kv, decode_times_kv = generate_kv(model=model, input_ids=prompt, max_new_tokens=max_new_tokens)

    end_kv.record()
    torch.cuda.synchronize()

    assert torch.equal(returned_tokens_base, returned_tokens_kv)

    print(f"Prompt ids: {prompt}")
    #print(f"Generated ids: {returned_tokens[:, prompt.shape[1]:]}")
    print(f"Prompt shape: {prompt.shape}")
    print(f"Final length: {returned_tokens_base.shape[1]}")
    #print(f"Full generation: {returned_tokens}")

    # Timing metrics for prefill and decode stages (baseline)
    print(f"---------Baseline---------")
    print(f"Prefill time: {prefill_time_base} ms")
    print(f"Total decode time: {decode_time_total_base} ms")
    print(f"Avg decode time: {decode_time_total_base / len(decode_times_base)}")
    print(f"Total generation time (with overhead): {start_base.elapsed_time(end_base):.2f} ms")

    # Timing metrics for prefill and decode stages with kv cache
    print(f"---------KV Cache---------")
    print(f"Prefill time: {prefill_time_kv} ms")
    print(f"Total decode time: {decode_time_total_kv} ms")
    print(f"Avg decode times: {decode_time_total_kv / len(decode_times_kv)}")
    print(f"Total generation time (with overhead): {start_kv.elapsed_time(end_kv):.2f} ms")

if __name__ == "__main__":
    main()
