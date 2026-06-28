"""
Дифференциальная приватность для локального обучения (DP-SGD).

Реализован канонический алгоритм DP-SGD (Abadi et al., 2016): для каждого
примера вычисляется индивидуальный градиент, его норма ограничивается порогом
C (clipping), к усреднённому градиенту добавляется гауссов шум с дисперсией
(σ·C)². Параметр σ (noise multiplier) задаёт уровень приватности; бюджет ε
оценивается RDP-аккаунтантом.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


@dataclass
class DPConfig:
    clip: float = 1.0          # порог клиппинга нормы градиента (C)
    sigma: float = 1.0         # множитель шума (noise multiplier)
    delta: float = 1e-5        # параметр δ для (ε, δ)-DP


def _example_loss(net, seq_i, pos_i, n_items, n_neg, device):
    neg = torch.randint(1, n_items, (1, n_neg), device=device)
    repr_vec = net.encode(seq_i)
    pos_s = (repr_vec * net.item_emb(pos_i)).sum(-1, keepdim=True)
    neg_s = (repr_vec.unsqueeze(1) * net.item_emb(neg)).sum(-1)
    return -F.logsigmoid(pos_s - neg_s).mean()


def dp_sgd_step(net, opt, seq, pos, n_items, n_neg, device, dp: DPConfig):
    """Один шаг DP-SGD по батчу: per-example клиппинг + гауссов шум."""
    params = [p for p in net.parameters() if p.requires_grad]
    accum = [torch.zeros_like(p) for p in params]
    B = len(seq)
    for i in range(B):
        opt.zero_grad(set_to_none=True)
        loss = _example_loss(net, seq[i:i + 1], pos[i:i + 1], n_items, n_neg, device)
        loss.backward()
        # норма градиента примера
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
        coef = min(1.0, dp.clip / (sq ** 0.5 + 1e-6))
        for j, p in enumerate(params):
            if p.grad is not None:
                accum[j] += p.grad.detach() * coef
    # усреднение + шум
    opt.zero_grad(set_to_none=True)
    for j, p in enumerate(params):
        noise = torch.normal(0.0, dp.sigma * dp.clip, size=p.shape, device=device)
        p.grad = (accum[j] + noise) / B
    opt.step()


def compute_epsilon(steps: int, sample_rate: float, sigma: float, delta: float = 1e-5):
    """Оценка бюджета приватности ε (RDP-аккаунтант Opacus, если установлен)."""
    if sigma <= 0:
        return float("inf")
    try:
        from opacus.accountants import RDPAccountant
        acc = RDPAccountant()
        for _ in range(int(steps)):
            acc.step(noise_multiplier=sigma, sample_rate=min(1.0, sample_rate))
        return float(acc.get_epsilon(delta=delta))
    except Exception:
        return None     # opacus не установлен — ε не подсчитан (укажем только σ)
