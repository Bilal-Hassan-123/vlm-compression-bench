import time, pandas as pd, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
print("Loading Qwen model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

# find whichever submodule BTP patched — don't guess the attribute path
btp = next(m for m in model.modules() if hasattr(m, "use_flash_pruning"))
print("BTP module:", type(btp).__name__)

df = pd.read_csv("chartqa_plain.csv").head(20)
print(f"Loaded {len(df)} rows")

predictions, start = [], time.time()
for i, row in df.iterrows():
    messages = [{"role": "user", "content": [
        {"type": "image", "image": row["image_path"]},
        {"type": "text", "text": str(row["question"]) + " Answer the question using a single word or phrase."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    # visual token span varies per image, so set it every row
    pos = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().squeeze(-1)
    btp.img_start_idx = pos[0]
    btp.img_num = pos.numel()

    with torch.inference_mode():
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=200)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    predictions.append(processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip())

    if (i + 1) % 50 == 0:
        el = time.time() - start
        print(f"{i+1}/{len(df)} done, {el:.1f}s, {el/(i+1):.2f}s/row")

df["prediction"] = predictions
df.to_csv("chartqa_btp_smoke.csv", index=False)
print("Saved chartqa_btp_smoke.csv")
