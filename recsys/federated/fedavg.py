"""
Федеративное усреднение (Federated Averaging, FedAvg).

Глобальная модель обучается без централизации сырых данных: на каждом раунде
выбирается подмножество клиентов (площадок), каждый локально дообучает копию
глобальной модели на своих сессиях, после чего сервер усредняет веса
пропорционально объёму данных клиента. Реализовано поверх последовательной
модели (GRU4Rec / SASRec) с BPR-функцией потерь и негативным сэмплированием.

При передаче dp-конфигурации локальное обучение выполняется приватно (DP-SGD,
см. recsys.federated.dp).
"""
from __future__ import annotations

import copy

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


def _bpr_loss(net, seq, pos, n_items, n_neg, device):
    neg = torch.randint(1, n_items, (len(seq), n_neg), device=device)
    repr_vec = net.encode(seq)
    pos_s = (repr_vec * net.item_emb(pos)).sum(-1, keepdim=True)
    neg_s = (repr_vec.unsqueeze(1) * net.item_emb(neg)).sum(-1)
    return -F.logsigmoid(pos_s - neg_s).mean()


def local_train(net, seqs, targets, *, epochs, batch_size, lr, n_neg, n_items,
                device, dp=None):
    """Локальное обучение клиента. Возвращает число выполненных шагов."""
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    seqs_t = torch.as_tensor(seqs, device=device)
    tgt_t = torch.as_tensor(targets, device=device)
    n = len(seqs_t)
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            seq, pos = seqs_t[idx], tgt_t[idx]
            if dp is None:
                loss = _bpr_loss(net, seq, pos, n_items, n_neg, device)
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                from recsys.federated.dp import dp_sgd_step
                dp_sgd_step(net, opt, seq, pos, n_items, n_neg, device, dp)
            steps += 1
    return steps


def _avg_state_dicts(states, weights):
    """Взвешенное усреднение state_dict нескольких клиентов."""
    w = np.asarray(weights, dtype=float); w = w / w.sum()
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        if avg[k].dtype.is_floating_point:
            stacked = torch.stack([s[k].float() * float(wi) for s, wi in zip(states, w)], dim=0)
            avg[k] = stacked.sum(dim=0).to(states[0][k].dtype)
        else:
            avg[k] = states[0][k]
    return avg


def federated_train(make_net, client_datasets, *, n_items, rounds=20,
                    clients_per_round=None, local_epochs=1, batch_size=256,
                    lr=1e-3, n_neg=4, device="cpu", dp=None, seed=42, verbose=True):
    """
    Запускает FedAvg. make_net() -> новая сеть; client_datasets: dict
    client -> (seqs, targets). Возвращает обученную глобальную сеть.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Не установлен torch")
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    global_net = make_net().to(device)
    clients = list(client_datasets.keys())
    k = clients_per_round or len(clients)
    total_steps = 0

    for r in range(rounds):
        chosen = list(rng.choice(clients, size=min(k, len(clients)), replace=False))
        states, sizes = [], []
        for c in chosen:
            seqs, targets = client_datasets[c]
            local = make_net().to(device)
            local.load_state_dict(copy.deepcopy(global_net.state_dict()))
            steps = local_train(local, seqs, targets, epochs=local_epochs,
                                batch_size=batch_size, lr=lr, n_neg=n_neg,
                                n_items=n_items, device=device, dp=dp)
            total_steps += steps
            states.append({kk: v.detach().clone() for kk, v in local.state_dict().items()})
            sizes.append(len(seqs))
        global_net.load_state_dict(_avg_state_dicts(states, sizes))
        if verbose:
            print(f"  [FedAvg] round {r + 1}/{rounds}  clients={len(chosen)}  "
                  f"avg_client_size={int(np.mean(sizes))}")
    global_net.eval()
    return global_net, total_steps
