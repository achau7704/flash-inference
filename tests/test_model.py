"""
Test functionality of model componenets

Usage:
python -m pytest tests/test_model.py -v

For print statements
python -m pytest -s tests/test_model.py -v
"""

from src.model.config import Config
from src.model.layernorm import LayerNorm
from src.model.mlp import MLP
from src.model.attention import MHAttention
from src.model.transformer import TransformerBlock, Transformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config(
        vocab_size=1024,
        hidden_size=1024,
        num_layers=12,
        num_heads=8,
        max_seq_len=10,
        max_batch_size=1,
        dtype=torch.float32,
        device=device
    )

def test_layernorm():
    module = LayerNorm(config=config)
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


def test_mlp():
    module = MLP(config=config)
    module.eval()

    # batch of 1, seq_len=4
    x = torch.ones(1, 4, config.hidden_size, dtype=config.dtype, device=config.device)
    output = module(x)

    assert x.shape == output.shape, "Shape mismatch between input and output for MLP layer"
    assert x.device == output.device, "Device mismatch between input and output for MLP layer"


def test_mhattention():
    module = MHAttention(config=config)
    module.eval()

    # batch of 1, seq_len=4
    x = torch.ones(1, 4, config.hidden_size, dtype=config.dtype, device=config.device)
    output = module(x)

    assert x.shape == output.shape, "Shape mismatch between input and output for MHAttention"
    assert x.device == output.device, "Device mismatch between input and output for MHAttention"


def test_transformer_block():
    module = TransformerBlock(config=config).eval()

    B, T, H = 1, 6, config.hidden_size
    t = 4  

    # random input
    x1 = torch.randn(B, T, H, dtype=config.dtype, device=config.device)
    x2 = x1.clone()

    # this basically makes sure the mask slicing works
    x2[:, t, :] += 1.0

    with torch.no_grad():
        out1 = module(x1)
        out2 = module(x2)

    # earlier outputs must not change (positions 0..t-1)
    assert torch.allclose(out1[:, :t, :], out2[:, :t, :], atol=1e-5, rtol=1e-5), \
        "Causality violated: earlier positions changed when a future token was perturbed"

    assert torch.isfinite(out1).all()
    assert not torch.allclose(out1, x1)
    assert x1.shape == out1.shape, "Shape mismatch between input and output for Transformer Block"
    assert x1.device == out1.device, "Device mismatch between input and output for Transformer Block"


def test_transformer_forward_basic():
    # Import your classes however your project is structured
    # e.g., from src.model.config import Config
    #       from src.model.transformer import Transformer
    from src.model.config import Config
    from src.model.transformer import Transformer

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # keep fp32 for tests; you can parametrize later

    # Tiny config for fast tests
    config = Config(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=32,
        max_batch_size=4,
        dtype=dtype,
        device=device,
    )

    model = Transformer(config).eval()

    B, T = 2, 12
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)

    # Shape: (B, T, vocab_size)
    assert logits.shape == (B, T, config.vocab_size)

    # Dtype + device
    assert logits.dtype == dtype
    assert logits.device == device

    # Finite outputs
    assert torch.isfinite(logits).all()


def test_transformer_is_deterministic_in_eval():
    from src.model.config import Config
    from src.model.transformer import Transformer

    torch.manual_seed(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    config = Config(
        vocab_size=64,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        max_seq_len=16,
        max_batch_size=2,
        dtype=dtype,
        device=device,
    )

    model = Transformer(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 10), device=device, dtype=torch.long)

    with torch.no_grad():
        out1 = model(input_ids)
        out2 = model(input_ids)

    # Exact equality should hold in eval for this pure module (no dropout)
    assert torch.equal(out1, out2)


def test_transformer_causality_smoke():
    """
    Smoke test: changing a token at position t should NOT change logits at positions < t.
    This catches causal mask bugs in attention.
    """
    from src.model.config import Config
    from src.model.transformer import Transformer

    torch.manual_seed(999)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    config = Config(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=32,
        max_batch_size=2,
        dtype=dtype,
        device=device,
    )

    model = Transformer(config).eval()

    B, T = 1, 20
    input_a = torch.randint(0, config.vocab_size, (B, T), device=device, dtype=torch.long)
    input_b = input_a.clone()

    # Pick a position t and change ONLY that token (and optionally some after it)
    t = 12
    input_b[0, t] = (input_b[0, t] + 1) % config.vocab_size

    with torch.no_grad():
        logits_a = model(input_a)
        logits_b = model(input_b)

    # Compare logits for positions strictly before t
    before_a = logits_a[:, :t, :]
    before_b = logits_b[:, :t, :]

    # Use a tolerance (GPU kernels can introduce tiny fp differences)
    torch.testing.assert_close(before_a, before_b, rtol=0, atol=0)
    # If you see occasional failures on GPU, relax slightly:
    # torch.testing.assert_close(before_a, before_b, rtol=1e-6, atol=1e-6)
