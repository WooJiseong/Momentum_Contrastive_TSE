# Vendored replacement for `asteroid.engine.optimizers.make_optimizer`.
# asteroid is intentionally NOT installed in the `pn-noisyflow` env (it pins an old torch/numpy and
# would clobber the cu128 stack). The MeanFlow config only ever asks for AdamW, so this 1:1 shim
# preserves the exact call site `make_optimizer(params, **config['optim'])` with config['optim'] =
# {optimizer: AdamW, lr: 1e-4, weight_decay: 0.01}.
import torch


def make_optimizer(params, optimizer="adam", lr=1e-3, weight_decay=0.0, **kwargs):
    """Build a torch optimizer. Mirrors the subset of asteroid's make_optimizer that MeanFlow uses."""
    name = str(optimizer).lower()
    betas = kwargs.get("betas", (0.9, 0.999))
    eps = kwargs.get("eps", 1e-8)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=kwargs.get("momentum", 0.0))
    raise ValueError(f"Unsupported optimizer '{optimizer}'. Use one of: adamw, adam, sgd.")
