"""
Эксперимент Трека 2: федеративное обучение и дифференциальная приватность.

Сравнивает три режима последовательной модели (GRU4Rec) на задаче Trivago:
  1) централизованное обучение (верхняя граница качества);
  2) федеративное обучение (FedAvg) по площадкам без приватности;
  3) федеративное обучение с дифференциальной приватностью (DP-SGD) при
     нескольких уровнях шума σ — с оценкой бюджета ε.

Главный результат — кривая trade-off «приватность (ε) ↔ качество (MRR)».

Запуск:
  python -m scripts.run_federated --data data/trivago --stream --nrows 2000000 --rounds 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from recsys.data.trivago import load, load_stream
from recsys.evaluation.metrics import MetricAccumulator
from recsys.evaluation.protocol import temporal_split
from recsys.federated.dp import DPConfig, compute_epsilon
from recsys.federated.fedavg import federated_train
from recsys.federated.partition import (build_client_datasets, partition_by_platform,
                                        partition_summary)

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def _evaluate(net, vocab, maxlen, device, test):
    """Оборачивает обученную сеть в ранкер и считает метрики на тесте."""
    from recsys.models.sequential.ranker import SequentialRanker
    r = SequentialRanker(kind="gru", maxlen=maxlen)
    r.vocab, r.net, r.device = vocab, net, device
    acc = MetricAccumulator(ks=(1, 5, 10))
    for inst in test:
        if inst.has_target:
            acc.add(r.rank(inst), inst.target)
    return acc.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trivago_sample")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--clients-per-round", type=int, default=8)
    ap.add_argument("--min-sessions", type=int, default=50)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=20)
    ap.add_argument("--sigmas", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    args = ap.parse_args()

    import torch
    from recsys.models.sequential.nets import build_net
    from recsys.models.sequential.seqdata import Vocab, build_sequences

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    t0 = time.time()

    data = (load_stream(args.data, max_rows=args.nrows) if args.stream
            else load(args.data, nrows=args.nrows))
    train, test = temporal_split(data.clickouts, test_frac=0.2)
    vocab = Vocab().fit(train)
    n_items = len(vocab)
    print(f"clickouts={len(data.clickouts)} train={len(train)} test={len(test)} items={n_items}")

    clients = partition_by_platform(train, min_sessions=args.min_sessions)
    client_ds = build_client_datasets(clients, vocab, args.maxlen)
    summ = partition_summary(client_ds)
    print(f"Клиентов (площадок): {summ['n_clients']}, последовательностей: {summ['total_sequences']}, "
          f"мин/макс на клиента: {summ['min']}/{summ['max']}")

    def make_net():
        return build_net("gru", n_items, args.emb_dim, args.maxlen)

    common = dict(n_items=n_items, rounds=args.rounds, local_epochs=args.local_epochs,
                  clients_per_round=args.clients_per_round, device=device)
    rows = []

    # --- 1) централизованное (один «клиент» = все данные) ---
    print("\n=== Централизованное обучение ===")
    all_seqs, all_tgt = build_sequences(train, vocab, args.maxlen)
    cnet, csteps = federated_train(make_net, {"all": (all_seqs, all_tgt)},
                                   n_items=n_items, rounds=args.rounds,
                                   local_epochs=args.local_epochs, clients_per_round=1, device=device)
    m = _evaluate(cnet, vocab, args.maxlen, device, test)
    rows.append({"mode": "Centralized", "sigma": None, "epsilon": None, **m})
    print(f"  MRR={m['MRR']:.4f}")

    # --- 2) федеративное без DP ---
    print("\n=== Федеративное обучение (FedAvg) ===")
    fnet, fsteps = federated_train(make_net, client_ds, **common)
    m = _evaluate(fnet, vocab, args.maxlen, device, test)
    rows.append({"mode": "Federated", "sigma": None, "epsilon": None, **m})
    print(f"  MRR={m['MRR']:.4f}")

    # --- 3) федеративное + DP при разных σ ---
    avg_client = max(1, summ["total_sequences"] // max(1, summ["n_clients"]))
    sample_rate = min(1.0, 256 / avg_client)
    for sigma in args.sigmas:
        print(f"\n=== Федеративное + DP (σ={sigma}) ===")
        dp = DPConfig(clip=1.0, sigma=sigma)
        dnet, dsteps = federated_train(make_net, client_ds, dp=dp, **common)
        eps = compute_epsilon(dsteps, sample_rate, sigma, dp.delta)
        m = _evaluate(dnet, vocab, args.maxlen, device, test)
        rows.append({"mode": f"Federated+DP", "sigma": sigma, "epsilon": eps, **m})
        print(f"  σ={sigma}  ε={eps}  MRR={m['MRR']:.4f}")

    payload = {"data": str(args.data), "n_clients": summ["n_clients"],
               "train": len(train), "test": len(test),
               "runtime_sec": round(time.time() - t0, 1), "results": rows}
    (RESULTS / "track2_federated.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print("\n=== Итог: privacy–quality trade-off ===")
    print(f"{'mode':<16}{'sigma':>7}{'epsilon':>10}{'MRR':>9}{'HitRate@5':>11}")
    for r in rows:
        eps = "-" if r["epsilon"] in (None,) else (f"{r['epsilon']:.2f}" if isinstance(r['epsilon'], float) else str(r['epsilon']))
        sg = "-" if r["sigma"] is None else f"{r['sigma']:.2f}"
        print(f"{r['mode']:<16}{sg:>7}{eps:>10}{r['MRR']:>9.4f}{r['HitRate@5']:>11.4f}")
    print(f"\nСохранено: {RESULTS / 'track2_federated.json'}  ({payload['runtime_sec']} c)")

    try:
        from scripts.plot_federated import plot_tradeoff
        plot_tradeoff(rows)
        print("График сохранён: results/track2_tradeoff.png")
    except Exception as e:  # pragma: no cover
        print(f"[viz] {e}")


if __name__ == "__main__":
    main()
