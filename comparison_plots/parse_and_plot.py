import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    out = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        denom = min(index + 1, window)
        out.append(running_sum / denom)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="comparison_plots/accuracy_plot.txt")
    parser.add_argument("--json", default="comparison_plots/accuracy.json")
    parser.add_argument("--output", default="report/CODET5_SFT_ACCURACY_PLOT_2.png")
    parser.add_argument("--window", type=int, default=7)
    args = parser.parse_args()

    input_path = Path(args.input)
    json_path = Path(args.json)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for line in input_path.read_text().splitlines():
        match = re.match(r"(\d+)/\d+ \| Acc: ([0-9.]+)", line.strip())
        if match:
            data.append(
                {
                    "step": int(match.group(1)),
                    "accuracy": float(match.group(2)),
                }
            )

    if not data:
        raise ValueError(f"No accuracy points found in {input_path}")

    json_path.write_text(json.dumps(data, indent=2))
    print(f"JSON file created at: {json_path}")

    steps = [d["step"] for d in data]
    accuracies = [d["accuracy"] for d in data]
    smoothed = moving_average(accuracies, args.window)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, accuracies, alpha=0.35, linewidth=1.5, label="Raw")
    plt.plot(steps, smoothed, linewidth=2.2, label=f"Moving Avg (window={args.window})")
    plt.xlabel("Examples Processed")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Examples For SFT Using CODET5 ")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
