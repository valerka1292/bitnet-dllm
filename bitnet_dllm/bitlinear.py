from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class STEQuantizeInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, Q_b: int) -> tuple[torch.Tensor, torch.Tensor]:
        eta = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8).detach()
        x_s = x * (Q_b / eta)
        x_q = x_s.round().clamp(-Q_b, Q_b - 1)
        ctx.save_for_backward(eta)
        ctx.Q_b = Q_b
        return x_q, eta

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _) -> tuple[torch.Tensor, None]:
        eta, = ctx.saved_tensors
        return grad_output * (ctx.Q_b / eta), None


class STEQuantizeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        gamma = weight.abs().mean().clamp(min=1e-8).detach()
        W_scaled = weight / gamma
        W_quant = W_scaled.round().clamp(-1, 1)
        ctx.save_for_backward(gamma)
        return W_quant, gamma

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _) -> torch.Tensor:
        return grad_output


class BitLinear(nn.Linear):
    """
    BitNet 1.58-bit ternary Linear layer.

    Weights quantized to {-1, 0, +1} via absmean scaling.
    Activations quantized per-token to activation_bits precision.

    Gradient note: ∂loss/∂W = grad_output · (1/gamma) where gamma is treated
    as a constant (detached). This is the unavoidable cost of normalizing weights
    before matmul. Compensate with reduced LR for BitLinear params (see trainer).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, activation_bits: int = 8):
        super().__init__(in_features, out_features, bias=bias)
        self.activation_bits  = activation_bits
        self.learnable_scale  = nn.Parameter(torch.ones(1))
        if self.bias is not None:
            self.bias._no_weight_decay = True

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        Q_b = 2 ** (self.activation_bits - 1)
        x_q, eta = STEQuantizeInput.apply(x, Q_b)
        return x_q, eta, Q_b

    def forward_quantized(self, x_q: torch.Tensor, eta: torch.Tensor, Q_b: int) -> torch.Tensor:
        W_q, gamma = STEQuantizeWeight.apply(self.weight)
        scale = eta * gamma / Q_b
        out = F.linear(x_q, W_q, None) * scale
        if self.bias is not None:
            out = out + self.bias
        return out * self.learnable_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, eta, Q_b = self.quantize_input(x)
        return self.forward_quantized(x_q, eta, Q_b)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.activation_bits}"
