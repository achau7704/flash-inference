"""
This script creates a config and a model, passes sample input tokens, runs the forward pass, and prints the output shape.
"""

from model.config import Config
import torch



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TestConfig = Config(
        vocab_size=1024,
        hidden_size=1024,
        num_layers=12,
        num_heads=8,
        max_seq_len=10,
        max_batch_size=1,
        dtype=torch.float32,
        device=device
    )
    print(TestConfig)
    x= TestConfig.head_dim
    print(x)

if __name__ == "__main__":
    main()
    
