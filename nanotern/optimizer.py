"""
MuonAdamW: Muon optimizer for 2D matrix params + AdamW for everything else.

Faithful single-GPU port of nanochat's optimizer
(nanochat-ref/nanochat/optim.py), stripped of the DDP communication paths.
Muon = Nesterov momentum + Polar Express orthogonalization (Newton-Schulz
variant) + NorMuon variance reduction + cautious weight decay.

References:
- https://github.com/KellerJordan/modded-nanogpt
- Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
- NorMuon: https://arxiv.org/pdf/2510.05491
"""
import torch
from torch import Tensor

from nanotern.common import COMPUTE_DTYPE


# -----------------------------------------------------------------------------
# AdamW fused kernel
# -----------------------------------------------------------------------------
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq,
                     step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    """
    Fused AdamW step. All math in fp32 (some params are stored in bf16),
    cast back at the end. 0-D CPU tensors avoid recompilation.
    """
    p32 = p.float()
    exp_avg32 = exp_avg.float()
    exp_avg_sq32 = exp_avg_sq.float()
    grad32 = grad.float()
    # decoupled weight decay
    p32.mul_(1 - lr_t * wd_t)
    # running averages
    exp_avg32.lerp_(grad32, 1 - beta1_t)
    exp_avg_sq32.lerp_(grad32.square(), 1 - beta2_t)
    # bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq32 / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p32.add_(exp_avg32 / denom, alpha=-step_size)
    # write back
    p.copy_(p32)
    exp_avg.copy_(exp_avg32)
    exp_avg_sq.copy_(exp_avg_sq32)


# -----------------------------------------------------------------------------
# Muon fused kernel
# -----------------------------------------------------------------------------
# Coefficients for Polar Express (num_iters=5, safety_factor=2e-2, cushion=2)
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    """
    Fused Muon step: Nesterov momentum -> MuonEq row equilibration ->
    Polar Express orthogonalization -> Muon+ renorm -> NorMuon variance
    reduction -> cautious weight decay + update.
    """
    # Nesterov momentum (stacked_grads is overwritten with the Nesterov g)
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # bf16 for speed (fp16 is unstable here)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    # MuonEq row equilibration: rescale each row to the mean row norm
    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X * (target / row_norm).to(X.dtype)

    # Polar Express orthogonalization
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):  # tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # wide matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X.to(stacked_params.dtype)

    # Muon+ renormalization: snap Frobenius norm to sqrt(min(m, n))
    target_norm = min(g.size(-2), g.size(-1)) ** 0.5
    current_norm = g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    g = g * (target_norm / current_norm).to(g.dtype)

    # NorMuon variance reduction (per-column adaptive scaling)
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """
    Combined Muon + AdamW optimizer, single-GPU only.
    param_groups must carry kind='muon' or kind='adamw' (see GPT.setup_optimizer).
    Muon groups: all params in a group must share one shape (caller's job).
    """
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def _compute_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p, p.grad, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    @torch.no_grad()
    def _compute_muon(self, group):
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        num_params = len(params)

        # group-level state lives under the first param
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        grad_chunk = torch.stack([pp.grad for pp in params])
        stacked_owned = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            grad_chunk, stacked_owned,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            group["ns_steps"], red_dim,
        )
        # copy updated stack back into individual params
        torch._foreach_copy_(params, list(stacked_owned[:num_params].unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._compute_adamw(group)
            elif group['kind'] == 'muon':
                self._compute_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")
