"""
Test functionality of KVCache class

Usage:
python -m pytest tests/test_kvcache.py -v

For print statements
python -m pytest -s tests/test_kvcache.py -v
"""

from src.model.config import Config
from src.runtime.kv_cache import KVCache
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_kvcache():
    module = KVCache(config=config, batch=1)
    module.eval()

    # batch of 1, seq_len=4
    x = torch.ones(1, 4, config.hidden_size, dtype=config.dtype, device=config.device)
    output = module(x)

    # add dimensions to beta and broadcast across batch and seq
    print(module.beta)
    print(module.beta.view(1, 1, -1))
    expected = module.beta.view(1, 1, -1).expand_as(output)
    print(expected)

    assert x.shape == output.shape, "Shape mismatch between input and output for LayerNorm"
    assert x.device == output.device, "Device mismatch between input and output for LayerNorm"
    torch.testing.assert_close(output, expected, msg="Accuracy error in LayerNorm")
