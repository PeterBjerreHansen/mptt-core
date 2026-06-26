import torch

# Keep CPU smoke tests fast and deterministic in constrained environments.
torch.set_num_threads(1)
