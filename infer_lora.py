"""
infer_lora.py

載入微調後的 LoRA Adapter，對指定的多輪對話情境進行推論驗證，
展示「情勒專家（情緒勒索大師）」在多輪對話下的回答效果。
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

# 測試用多輪對話劇本（測試至少 2~3 輪）
TEST_DIALOGUES = [
    {
        "title": "情境一：伴侶臨時失約去聚餐",
        "turns": [
            "我今天晚上要跟同事去聚餐，不能陪妳吃晚餐了。",
            "妳不要這樣嘛，改天我再請妳吃大餐補償好不好？",
            "我真的沒有勉強，妳是我的第一順位啊！",
        ],
    },
    {
        "title": "情境二：拒絕朋友借錢",
        "turns": [
            "兄弟，我最近手頭有點緊，沒辦法借你五萬塊應急。",
            "我們是很好的朋友，但我的存款真的要留著繳房貸，萬一違約很麻煩。",
            "真不是錢的問題，我有我的經濟負擔，你不能這樣情勒我啊！",
        ],
    },
    {
        "title": "情境三：下班拒絕主管臨時加班",
        "turns": [
            "主管，現在已經晚上七點了，我等下有私人家聚，這份報告可以週一再交嗎？",
            "主管，合約上本來就是寫週一交付，客戶現在是臨時要求的！",
        ],
    },
]


def generate_response(model, tokenizer, messages):
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    eos_token_ids = [tokenizer.eos_token_id, 151645]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
        )

    # 擷取生成部分
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def main():
    if not LORA_DIR.is_dir():
        raise FileNotFoundError(f"找不到 LoRA Adapter 目錄：{LORA_DIR.resolve()}，請先執行 train_lora.py！")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"載入基底模型: {BASE_MODEL_ID} (裝置: {device})...")
    tokenizer = AutoTokenizer.from_pretrained(str(LORA_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"掛載 LoRA Adapter: {LORA_DIR.resolve()} ...")
    model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
    model.eval()

    print("\n" + "=" * 72)
    print("【情勒專家（情緒勒索大師）多輪對話驗證展示】")
    print("=" * 72)

    output_lines = []

    for dialog in TEST_DIALOGUES:
        title = dialog["title"]
        print(f"\n### {title}\n")
        output_lines.append(f"\n### {title}\n")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for round_idx, user_input in enumerate(dialog["turns"], 1):
            messages.append({"role": "user", "content": user_input})
            print(f"你 (第 {round_idx} 輪): {user_input}")
            output_lines.append(f"你 (第 {round_idx} 輪): {user_input}")

            reply = generate_response(model, tokenizer, messages)
            print(f"情勒專家: {reply}\n")
            output_lines.append(f"情勒專家: {reply}\n")

            messages.append({"role": "assistant", "content": reply})

        print("-" * 72)

    # 儲存展示結果供 README 引用
    with open("outputs/guilt_trip_lora/sample_dialogue.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"多輪對話成果已匯出至: outputs/guilt_trip_lora/sample_dialogue.txt")


if __name__ == "__main__":
    main()
