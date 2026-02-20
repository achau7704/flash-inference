"""
Script that runs baseline prefill and decode.
"""

import torch
from src.model.config import Config
from src.model.transformer import Transformer
from src.runtime.generate_baseline import generate_baseline


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
        max_seq_len=512,
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
    max_new_tokens = 500

    # warmup block
    # without warmup we measure starup overhead like CUDA context init etc.
    warmup_runs = 3
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(prompt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # create CUDA events with timing enabled
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # call the generate function
    returned_tokens, prefill_time, decode_time_total, decode_times = generate_baseline(model=model, input_ids=prompt, max_new_tokens=max_new_tokens)

    end.record()

    # wait for all queued GPU work to finish before querying time on CPU
    torch.cuda.synchronize()

    print(f"Prompt ids: {prompt}")
    print(f"Generated ids: {returned_tokens[:, prompt.shape[1]:]}")
    print(f"Prompt shape: {prompt.shape}")
    print(f"Final length: {returned_tokens.shape[1]}")
    print(f"Full generation: {returned_tokens}")

    # Timing metrics for prefill and decode stages
    print(f"Prefill time: {prefill_time} ms")
    print(f"Total decode time: {decode_time_total} ms")
    print(f"Decode times: {decode_times}")
    print(f"Total generation time (with overhead): {start.elapsed_time(end):.2f} ms")

if __name__ == "__main__":
    main()
