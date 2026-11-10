"""
GPT model: a from-scratch implementation of the nanochat architecture.

Spec source: /home/albert/git/nanochat-ref/nanochat/gpt.py (read as a
specification, reimplemented here; see DECISIONS.md D2).

Features:
- rotary embeddings (no learned positional embeddings), RoPE base 100000
- QK normalization (+1.2 attention sharpness scale)
- untied wte / lm_head
- relu^2 MLP (4x width), no bias anywhere
- parameterless RMSNorm
- value embeddings (ResFormer) on alternating layers, input-gated
- "smear": mix previous token embedding into current token
- "backout": subtract a cached mid-layer residual before the final norm
- per-layer resid_lambdas / x0_lambdas
- sliding window attention ("SSSL" pattern; S = quarter context)
- logit softcap (15)

Single-GPU only. Attention via SDPA: is_causal flash kernel for full-context
layers, explicit boolean mask for sliding-window layers (no extra packages).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanotern.common import COMPUTE_DTYPE


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


# Config presets (see DECISIONS.md D3 for the parameter accounting)
CONFIG_PRESETS = {
    # ~37M params, CPU smoke test
    "tiny": dict(n_layer=4, n_head=2, n_kv_head=2, n_embd=256),
    # ~126M params, GPU smoke test (GPT-2 class)
    "smoke": dict(n_layer=8, n_head=4, n_kv_head=4, n_embd=512),
    # ~1.10B params, the target model
    "tern1b": dict(n_layer=21, n_head=11, n_kv_head=11, n_embd=1408),
}


def make_config(preset: str = "tern1b", **overrides) -> GPTConfig:
    kwargs = dict(CONFIG_PRESETS[preset])
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts its weight to the input dtype in forward.
    Master weights stay fp32 for optimizer precision, matmuls run in the
    activation dtype (typically bf16)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Value Embedding on alternating layers; last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    # rotates by -theta (transpose of the textbook convention); functionally
    # equivalent since only relative q/k rotation matters (same as nanochat)
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q: (B, H, Tq, D); k, v: (B, H_kv, Tk, D).
    window_size: (left, right); left = -1 for full context.
    (Ported from nanochat's SDPA fallback, which handles the non-FA3 case.)
    """
    Tq, Tk = q.size(2), k.size(2)
    window = window_size[0]

    # full context, same length -> flash kernel path
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # otherwise build an explicit boolean mask (training with windows, or chunked prefill)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # value embedding gate (input-dependent, per head), from the first 12 channels
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) \
            if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # value residual (ResFormer): mix in value embedding with input-dependent gate
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # rotary embeddings + QK norm
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        # attention (SDPA with optional sliding window, optional KV cache)
        if kv_cache is None:
            enable_gqa = self.n_head != self.n_kv_head
            y = _sdpa_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                window_size, enable_gqa).transpose(1, 2)
        else:
            k_cache, v_cache, pos = kv_cache[self.layer_idx]
            end_pos = pos + T
            k_cache[:, pos:end_pos, :, :] = k
            v_cache[:, pos:end_pos, :, :] = v
            enable_gqa = self.n_head != self.n_kv_head
            y = _sdpa_attention(q.transpose(1, 2),
                                k_cache[:, :end_pos, :, :].transpose(1, 2),
                                v_cache[:, :end_pos, :, :].transpose(1, 2),
                                window_size, enable_gqa).transpose(1, 2)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class KVCache:
    """
    Minimal KV cache for single-batch autoregressive inference.
    Per-layer (k_cache, v_cache) of shape (B=1, T_max, H_kv, D), shared pos.
    Also carries the previous token embedding for the smear at decode time.
    """
    def __init__(self, n_layers, n_kv_head, head_dim, T_max, device, dtype):
        self.n_layers = n_layers
        self.k_cache = [torch.empty(1, T_max, n_kv_head, head_dim, device=device, dtype=dtype)
                        for _ in range(n_layers)]
        self.v_cache = [torch.empty(1, T_max, n_kv_head, head_dim, device=device, dtype=dtype)
                        for _ in range(n_layers)]
        self.pos = torch.zeros(1, dtype=torch.int32, device=device)
        self.prev_embedding = None

    def __getitem__(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx], self.pos

    def advance(self, T):
        self.pos += T

    def get_pos(self):
        return int(self.pos.item())


class GPT(nn.Module):
    def __init__(self, config):
        """
        NOTE: this __init__ may run under a meta device context (shapes/dtypes
        only). All data initialization happens in init_weights().
        """
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        # pad vocab for tensor-core friendliness (crop in forward)
        pad_vocab_size_to = 64
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = padded_vocab_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # per-layer learnable scalars
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # smear: mix previous token's embedding into current token
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # backout: subtract cached mid-layer residual before final norm
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # value embeddings (ResFormer), alternating layers
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # rotary embeddings (fake init here; real in init_weights)
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialization (same as nanochat):
        wte: normal std 0.8; lm_head: normal std 0.001
        q/k/v: uniform std 1/sqrt(n_embd); c_proj (attn): zeros
        c_fc: uniform std 0.4/sqrt(n_embd); c_proj (mlp): zeros
        resid_lambdas: 1.15 -> 1.05 (decaying); x0_lambdas: 0.20 -> 0.05
        smear_lambda: 0; backout_lambda: 0.2; gates: uniform [0, 0.02]
        value_embeds: uniform std 1/sqrt(n_embd)
        """
        d, l = self.config.n_embd, self.config.n_layer
        for i in range(l):
            mat = d ** -0.5
            self.transformer.h[i].attn.c_q.weight.uniform_(-mat, mat)
            self.transformer.h[i].attn.c_k.weight.uniform_(-mat, mat)
            self.transformer.h[i].attn.c_v.weight.uniform_(-mat, mat)
            self.transformer.h[i].attn.c_proj.weight.zero_()
            self.transformer.h[i].mlp.c_fc.weight.uniform_(-0.4 * d**-0.5, 0.4 * d**-0.5)
            self.transformer.h[i].mlp.c_proj.weight.zero_()
            if self.transformer.h[i].attn.ve_gate is not None:
                self.transformer.h[i].attn.ve_gate.weight.uniform_(0.0, 0.02)
        # token embedding and lm_head
        self.transformer.wte.weight.normal_(0, 0.8)
        self.lm_head.weight.normal_(0, 0.001)
        # value embeddings
        for ve in self.value_embeds.values():
            ve.weight.uniform_(-d**-0.5, d**-0.5)
        # learnable scalars
        resid_init = [1.15 - 0.10 * i / (l - 1) for i in range(l)]
        self.resid_lambdas.data.copy_(torch.tensor(resid_init))
        x0_init = [0.20 - 0.15 * i / (l - 1) for i in range(l)]
        self.x0_lambdas.data.copy_(torch.tensor(x0_init))
        self.smear_lambda.data.zero_()
        self.backout_lambda.data.fill_(0.2)
        # re-init rotary embeddings with real dtype (fp32)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        # embeddings are stored in the compute dtype (low precision is OK for them)
        self.transformer.wte.to(COMPUTE_DTYPE)
        for ve in self.value_embeds.values():
            ve.to(COMPUTE_DTYPE)
        # linear layers stay fp32 (master precision)
        for module in self.modules():
            if isinstance(module, Linear):
                module.to(dtype=torch.float32)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # meta-safe version (returns fake tensors with correct shape/dtype if on meta)
        device = device if device is not None else self.transformer.wte.weight.device
        r = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        outer = torch.outer(r, freqs)
        cos, sin = outer.cos(), outer.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert set(pattern) <= {"L", "S"}
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA tile size
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)  # final layer always gets full context
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def num_matmul_params(self):
        """Params that participate in matmuls with the token stream (all Linear weights)."""
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel()
                   + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def estimate_flops(self):
        """Estimated FLOPs per token for training (forward + backward)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * self.num_matmul_params() + attn_flops

    def setup_optimizer(self, unembedding_lr=0.008, embedding_lr=0.3, matrix_lr=0.02,
                        weight_decay=0.0, scalar_lr=0.5):
        """
        Muon for matrix params + AdamW for the rest (see optimizer.py).
        AdamW LRs scaled by 1/sqrt(dmodel/768), tuned for 768-dim models.
        """
        from nanotern.optimizer import MuonAdamW
        model_dim = self.config.n_embd

        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(embedding_params) + len(lm_head_params)
            + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)
        )

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr,
                 betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # grab the rotary embeddings for the current sequence length
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx on different devices"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # embed the tokens
        x = self.transformer.wte(idx)  # (B, T, C)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # smear: mix previous token's embedding into current position
        if kv_cache is None:
            # T==1 is legal for prompt-free generation; only training (with
            # targets) requires a sequence to shift against.
            assert T > 1 or targets is None, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # forward the trunk
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # lm_head (compute logits)
        softcap = 15
        logits = self.lm_head(x)  # (B, T, padded_vocab_size)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
            return loss
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference (batch size 1, no KV cache:
        recompute each step). tokens is a Python list of ints; yields ints.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
