"""
This script generates max_new_tokens tokens given a model and a starting prompt.

input_ids is the prompt.

I also measure the time for the prefill phase (process the prompt) and decode phase (each subsequent iteration).
"""

import torch

def generate(model, input_ids, max_new_tokens):

    # make sure model and input_ids on same device
    assert input_ids.device == next(model.parameters()).device      # model.parameters() is an iterator so next gets the first parameter tensor

    prefill_time
    decode_time_total

    model.eval()
    with torch.no_grad():
        for i in range(max_new_tokens):
            # run the forward pass
            logits = model(input_ids)
            last_logits = logits[:, -1, :]

            # greedy decoding
            next_id = torch.argmax(last_logits, dim=-1)
            assert next_id.dtype == torch.long

            # change shape of next_id from (B,) to (B, 1)
            next_id = next_id.unsqueeze(dim=-1)

            # append new token to input_ids
            input_ids = torch.cat((input_ids, next_id), dim=1)

            if input_ids.shape[1] >= model.max_seq_len:
                break

    # return the generated tokens
    return input_ids, prefill_time, decode_time_total
