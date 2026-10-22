"""
plot_loss.py

讀取 outputs/guilt_trip_lora/trainer_state.json，繪製訓練與驗證 Loss 曲線，
並輸出為 outputs/guilt_trip_lora/loss_curve.png。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs/guilt_trip_lora")
STATE_FILE = OUTPUT_DIR / "trainer_state.json"
PLOT_FILE = OUTPUT_DIR / "loss_curve.png"


def main():
    if not STATE_FILE.is_file():
        raise FileNotFoundError(f"找不到：{STATE_FILE.resolve()}，請先執行 train_lora.py！")

    with STATE_FILE.open("r", encoding="utf-8") as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    if not logs:
        raise RuntimeError("trainer_state.json 中沒有記錄 log_history。")

    train_loss = [(item["step"], item["loss"]) for item in logs if "step" in item and "loss" in item]
    valid_loss = [(item["step"], item["eval_loss"]) for item in logs if "step" in item and "eval_loss" in item]

    plt.figure(figsize=(9, 5))
    if train_loss:
        x, y = zip(*train_loss)
        plt.plot(x, y, label="Training Loss", color="#1f77b4", linewidth=2)
    if valid_loss:
        x, y = zip(*valid_loss)
        plt.plot(x, y, marker="o", label="Validation Loss", color="#ff7f0e", linewidth=2)

    plt.xlabel("Optimizer Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Guilt-Trip Expert LoRA Fine-Tuning Loss Curve", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()
    print(f"訓練 Loss 曲線已儲存至: {PLOT_FILE.resolve()}")


if __name__ == "__main__":
    main()
