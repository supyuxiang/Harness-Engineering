import argparse
import csv
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt

from llm_client import call_llm as _raw_call_llm, count_tokens, count_messages_tokens, truncate_to_tokens
from solution import MyHarness


def make_controlled_llm(max_prompt_tokens: int, tracker: dict, lock: threading.Lock):
    def _call(messages: list[dict]) -> str:
        prompt_text = " ".join(m.get("content", "") for m in messages)
        n = count_tokens(prompt_text)
        if n > max_prompt_tokens:
            messages = list(messages)
            excess = n - max_prompt_tokens
            for i in range(len(messages) - 1, -1, -1):
                if excess <= 0:
                    break
                content = messages[i].get("content", "")
                msg_tokens = count_tokens(content)
                if msg_tokens <= excess:
                    messages[i] = {**messages[i], "content": ""}
                    excess -= msg_tokens
                else:
                    messages[i] = {**messages[i], "content": truncate_to_tokens(content, msg_tokens - excess)}
                    excess = 0
            truncated_by = n - max_prompt_tokens
            n = count_tokens(" ".join(m.get("content", "") for m in messages))
            print(f"[WARNING] prompt truncated by {truncated_by} tokens (budget={max_prompt_tokens})", file=sys.stderr)
        resp = _raw_call_llm(messages)
        with lock:
            tracker["prompt"] += n
            tracker["completion"] += count_tokens(resp)
        return resp

    return _call


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_one_k(train, dev, top_k: int, runs: int, workers: int, max_prompt_tokens: int):
    run_accuracies = []
    run_details = []

    for run_idx in range(runs):
        tracker = {"prompt": 0, "completion": 0}
        lock = threading.Lock()
        llm = make_controlled_llm(max_prompt_tokens, tracker, lock)

        harness = MyHarness(
            llm,
            count_tokens,
            count_messages_tokens,
            max_prompt_tokens,
            top_k=top_k,
        )
        for item in train:
            harness.update(item["text"], item["label"])

        predictions = [None] * len(dev)
        error_log = []
        t0 = time.time()

        def run_one(idx_item):
            idx, item = idx_item
            try:
                pred = harness.predict(item["text"])
                return idx, pred.strip(), None
            except Exception as exc:  # keep parity with run.py behavior
                return idx, "", str(exc)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_one, (i, item)) for i, item in enumerate(dev)]
            done = 0
            for fut in as_completed(futures):
                idx, pred, err = fut.result()
                predictions[idx] = pred
                if err:
                    error_log.append((idx, err))
                done += 1
                sys.stdout.write(
                    f"\r[k={top_k:>2}] run {run_idx + 1}/{runs} progress: {done}/{len(dev)}"
                )
                sys.stdout.flush()
        print()

        correct = sum(1 for item, pred in zip(dev, predictions) if pred == item["label"])
        acc = correct / len(dev)
        elapsed = time.time() - t0
        run_accuracies.append(acc)
        run_details.append(
            {
                "run_idx": run_idx + 1,
                "acc": acc,
                "elapsed_s": elapsed,
                "errors": len(error_log),
                "prompt_per_sample": tracker["prompt"] / len(dev),
                "completion_per_sample": tracker["completion"] / len(dev),
            }
        )
        print(
            f"[k={top_k:>2}] run {run_idx + 1}/{runs}: "
            f"acc={acc:.4f}, elapsed={elapsed:.1f}s, errors={len(error_log)}"
        )

    avg_acc = sum(run_accuracies) / len(run_accuracies)
    return avg_acc, run_details


def save_results_csv(path: str, summary_rows: list[dict], detail_rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "k", "avg_acc", "run_idx", "run_acc", "elapsed_s", "errors", "prompt_per_sample", "completion_per_sample"])
        for row in summary_rows:
            writer.writerow(["summary", row["k"], row["avg_acc"], "", "", "", "", "", ""])
        for row in detail_rows:
            writer.writerow(
                [
                    "detail",
                    row["k"],
                    "",
                    row["run_idx"],
                    row["acc"],
                    row["elapsed_s"],
                    row["errors"],
                    row["prompt_per_sample"],
                    row["completion_per_sample"],
                ]
            )


def plot_curve(path: str, ks: list[int], avg_accs: list[float]):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.plot(ks, avg_accs, marker="o", linewidth=2.2, markersize=6, color="#2a6fdb")
    ax.fill_between(ks, avg_accs, [min(avg_accs)] * len(avg_accs), alpha=0.12, color="#2a6fdb")

    best_idx = max(range(len(avg_accs)), key=lambda i: avg_accs[i])
    best_k = ks[best_idx]
    best_acc = avg_accs[best_idx]
    ax.scatter([best_k], [best_acc], color="#d62828", s=70, zorder=3)
    ax.annotate(
        f"best: k={best_k}, acc={best_acc:.4f}",
        xy=(best_k, best_acc),
        xytext=(best_k, best_acc + 0.01),
        ha="center",
        fontsize=10,
        color="#d62828",
        arrowprops=dict(arrowstyle="->", color="#d62828", lw=1.2),
    )

    ax.set_title("Top-k Sweep on DEV (runs=4 average)", fontsize=14, pad=12)
    ax.set_xlabel("top-k", fontsize=12)
    ax.set_ylabel("average accuracy", fontsize=12)
    ax.set_xticks(ks)
    ax.set_ylim(bottom=max(0.0, min(avg_accs) - 0.03), top=min(1.0, max(avg_accs) + 0.03))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train_dev.jsonl")
    parser.add_argument("--dev", default="data/test_dev.jsonl")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--ks",
        type=str,
        default="5,10,15,20,25,30,35,40,45,50,55,60",
    )
    parser.add_argument("--csv-out", default="topk_sweep_results.csv")
    parser.add_argument("--plot-out", default="topk_sweep_plot.png")
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    train = load_jsonl(args.train)
    dev = load_jsonl(args.dev)

    print("=" * 68)
    print("Top-k Sweep")
    print(f"Train={len(train)}  Dev={len(dev)}  runs={args.runs}  workers={args.workers}")
    print(f"K list: {ks}")
    print("=" * 68)

    summary_rows = []
    detail_rows = []

    global_t0 = time.time()
    for k in ks:
        print(f"\n>>> Evaluating k={k}")
        avg_acc, details = evaluate_one_k(
            train=train,
            dev=dev,
            top_k=k,
            runs=args.runs,
            workers=args.workers,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        summary_rows.append({"k": k, "avg_acc": avg_acc})
        for d in details:
            detail_rows.append({"k": k, **d})
        print(f"[k={k}] avg_acc={avg_acc:.4f}")

    elapsed = time.time() - global_t0
    ks_sorted = [r["k"] for r in summary_rows]
    avg_accs = [r["avg_acc"] for r in summary_rows]

    save_results_csv(args.csv_out, summary_rows, detail_rows)
    plot_curve(args.plot_out, ks_sorted, avg_accs)

    print("\n" + "=" * 68)
    print("Sweep complete.")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"CSV:  {args.csv_out}")
    print(f"Plot: {args.plot_out}")
    best_idx = max(range(len(avg_accs)), key=lambda i: avg_accs[i])
    print(f"Best: k={ks_sorted[best_idx]}, avg_acc={avg_accs[best_idx]:.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
