import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare runs in results/ on the validation data (no test accuracy)"
    )
    parser.add_argument("--dataset", default="mnist", help="only runs on this data set")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    return parser.parse_args()


def load_results(root="results"):
    results = []
    for path in sorted(Path(root).glob("*/result.json")):
        result = json.loads(path.read_text())
        result["run"] = path.parent.name
        results.append(result)
    return results


def to_row(result):
    config = result["config"]
    best_epoch = result["best_epoch"]
    # without validation there is no best epoch, the last epoch's weights are kept
    train_accuracy = result["train_accuracy"][best_epoch - 1 if best_epoch else -1]
    activations = config["activations"]
    return {
        "run": result["run"],
        "by": result["run_by"],
        "topology": "-".join(str(n) for n in config["topology"]),
        "activations": activations[0] if len(set(activations)) == 1 else ",".join(activations),
        "optimizer": config["optimizer_type"],
        "parameters": _format_count(result["parameter_count"]),
        "best epoch": f"{best_epoch}/{config['epoch_count']}" if best_epoch else "–",
        "train": _format_percent(train_accuracy),
        "val": _format_percent(result["best_validation_accuracy"]),
        "time": f"{result['duration_seconds']:.0f} s",
    }


def print_table(rows, markdown=False):
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(row[c]) for row in rows)) for c in columns}
    right_aligned = {"parameters", "best epoch", "train", "val", "time"}

    def cell(column, text):
        if column in right_aligned:
            return text.rjust(widths[column])
        return text.ljust(widths[column])

    if markdown:
        print("| " + " | ".join(cell(c, c) for c in columns) + " |")
        print("|" + "|".join(
            "-" * (widths[c] + 1) + ":" if c in right_aligned else "-" * (widths[c] + 2)
            for c in columns
        ) + "|")
        for row in rows:
            print("| " + " | ".join(cell(c, row[c]) for c in columns) + " |")
    else:
        print("  ".join(cell(c, c) for c in columns))
        print("  ".join("─" * widths[c] for c in columns))
        for row in rows:
            print("  ".join(cell(c, row[c]) for c in columns))


def main():
    args = parse_args()
    results = [
        r for r in load_results() if r["config"]["data_set_type"] == args.dataset
    ]
    if not results:
        print(f"No runs for data set '{args.dataset}' in results/")
        return

    # best validation accuracy first, runs without validation last
    results.sort(key=lambda r: r["best_validation_accuracy"] or -1, reverse=True)
    print_table([to_row(r) for r in results], markdown=args.markdown)
    print()
    print("train = train accuracy of the kept weights (best epoch, or last without validation)")


def _format_percent(value):
    return "–" if value is None else f"{value:.2%}"


def _format_count(value):
    return "–" if value is None else f"{value:,}".replace(",", " ")


if __name__ == "__main__":
    main()
