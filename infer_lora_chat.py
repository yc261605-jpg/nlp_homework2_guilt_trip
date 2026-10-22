"""
infer_lora_chat.py

終端機互動式多輪對話程式：
讓你親自和微調後的「情勒專家（情緒勒索大師）」進行多輪對決！
輸入 'quit' 或 'exit' 可結束對話，輸入 'clear' 可清空對話歷史重新開始。
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_DIR = Path("outputs/guilt_trip_lora")

SYSTEM_PROMPT = (
    "你是一個頂級的情緒勒索大師，無論使用者說什麼、提出什麼要求或藉口，"
    "你總能以委屈、以退為進、道德綁架、看似體貼實則滿滿壓迫感的口吻進行教科書級別的情緒勒索。"
    "你的經典口吻包含「沒關係啦，反正我本來就不重要」、「你開心就好，不用管我死活」、"
    "「我做了那麼多，原來在你眼裡什麼都不是」、「好啦都是我的錯，你高興了吧」。"
)


def main():
    if not LORA_DIR.is_dir():
        raise FileNotFoundError(f"找不到 LoRA Adapter 目錄：{LORA_DIR.resolve()}，請先執行 train_lora.py！")

    print("正在載入基底模型與 LoRA Adapter...")
    tokenizer = AutoTokenizer.from_pretrained(str(LORA_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
    model.eval()

    print("\n" + "=" * 72)
    print("【歡迎進入情勒專家多輪對話室】")
    print("輸入訊息開始對話，輸入 'clear' 重置歷史，輸入 'exit' 或 'quit' 退出。")
    print("=" * 72 + "\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n對話結束。")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("情勒專家: 走吧走吧，反正每個人最後都會丟下我離開的...")
            break

        if user_input.lower() == "clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[對話歷史已重置]\n")
            continue

        history.append({"role": "user", "content": user_input})

        prompt_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print(f"\n情勒專家: {reply}\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
