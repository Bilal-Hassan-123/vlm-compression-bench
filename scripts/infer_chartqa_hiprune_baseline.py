import os
import time
import pandas as pd
import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# Full retention = no pruning. LLaVA-1.5's CLIP ViT-L/14@336px produces 576 visual tokens,
# so retention=576 keeps all of them -> true uncompressed baseline on this exact pipeline.
os.environ["HIPRUNE_RETENTION"] = "576"
os.environ["HIPRUNE_ALPHA"] = "0.1"
os.environ["HIPRUNE_OBJECT_LAYER"] = "9"

model_path = "liuhaotian/llava-v1.5-7b"
conv_mode = "llava_v1"

print("Loading model...")
disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

df = pd.read_csv("chartqa_plain.csv")
print(f"Loaded {len(df)} rows")

predictions = []
start = time.time()

for i, row in df.iterrows():
    image = Image.open(row["image_path"]).convert("RGB")
    qs = DEFAULT_IMAGE_TOKEN + "\n" + str(row["question"]) + " Answer the question using a single word or phrase."

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids, v_token_num, image_attns = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=200,
            use_cache=True,
        )

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    predictions.append(answer)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"{i+1}/{len(df)} done, {elapsed:.1f}s elapsed, {elapsed/(i+1):.2f}s/row, v_tokens={v_token_num}")

df["prediction"] = predictions
df.to_csv("chartqa_predictions_hiprune_baseline.csv", index=False)
print("Saved chartqa_predictions_hiprune_baseline.csv")
