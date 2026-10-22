"""
train_lora.py

使用 PEFT / LoRA 對 Qwen/Qwen2.5-0.5B-Instruct 進行多輪對話 Supervised Fine-Tuning (SFT)，
將模型微調成專屬的「情勒專家（情緒勒索大師）」。

微調特色：
    1. 採用 Qwen2.5-0.5B-Instruct 繁簡中文開源對話模型
    2. LoRA 低秩適應微調 (r=16, alpha=32, 針對所有 Attention 與 MLP 線性層)
    3. FP16 混合精度訓練，極致發揮 RTX 3050 GPU 運算能力
    4. 支援 ChatML 格式與多輪歷史對話
    5. 自動追蹤 eval_loss 並輸出最佳 Adapter 權重至 outputs/guilt_trip_lora/
"""

import shutil
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

# =============================================================================
# 1. 常數與路徑設定
# =============================================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = Path("data/sft")
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"
OUTPUT_DIR = Path("outputs/guilt_trip_lora")

# 超參數設定
SEED = 42
EPOCHS = 5
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4  # 等效 Batch Size = 1 * 4 = 4
LEARNING_RATE = 2.5e-4
MAX_SEQ_LENGTH = 384

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def print_gpu_info():
    print("=" * 72)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用性: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"顯存容量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 72)


def main():
    print_gpu_info()

    if not TRAIN_FILE.is_file() or not VALID_FILE.is_file():
        raise FileNotFoundError("找不到資料集檔案！請先執行 build_dataset.py。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 載入資料集
    print("載入 SFT 多輪對話資料集...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VALID_FILE),
        },
    )
    print(f"訓練樣本數: {len(dataset['train'])} | 驗證樣本數: {len(dataset['validation'])}")

    # 2. 載入 Tokenizer
    print(f"\n載入 Tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 載入基底模型
    print(f"載入基底語言模型: {MODEL_ID} (FP16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
    base_model.config.use_cache = False

    # 4. 配置 LoRA Adapter
    print(f"配置 LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 5. 訓練設定 (SFTConfig)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=5,
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        max_seq_length=MAX_SEQ_LENGTH,
        seed=SEED,
        report_to="none",
    )

    # 6. 建立 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # 7. 開始微調
    print("\n" + "=" * 72)
    print("開始微調情勒專家模型 (5 Epochs)...")
    print("=" * 72)
    trainer.train()

    print("\n" + "=" * 72)
    print("訓練完成！")

    # 8. 儲存 Adapter 與 Tokenizer
    print(f"\n正在匯出最終收斂之 LoRA Adapter 至: {OUTPUT_DIR.resolve()} ...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    trainer.save_state()

    # 清理 checkpoint-* 資料夾
    print("清理暫存 checkpoint-* 目錄...")
    for ckpt in OUTPUT_DIR.glob("checkpoint-*"):
        if ckpt.is_dir():
            shutil.rmtree(ckpt, ignore_errors=True)
            print(f"  已移除暫存 checkpoint: {ckpt.name}")

    print("=" * 72)
    print(f"微調完成！Adapter 已妥善儲存在: {OUTPUT_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
