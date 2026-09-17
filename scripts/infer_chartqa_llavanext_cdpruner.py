import os
import time
import pandas as pd
import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

VISUAL_TOKEN_NUM = 192  # same total budget, now enforced across all AnyRes tiles

model_path = "liuhaotian/llava-v1.6-vicuna-7b"
conv_mode = "llava_v1"  # TODO verify - see grep below

print("Loading model...")
disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, visual_token_num=VISUAL_TOKEN_NUM
)

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
        output_ids, visual_token_num = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            texts=[qs],
            do_sample=False,
            max_new_tokens=200,
            use_cache=True,
        )
        if hasattr(model.model, 'visual_token_num'):
            visual_token_num = model.model.visual_token_num

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    predictions.append(answer)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"{i+1}/{len(df)} done, {elapsed:.1f}s elapsed, {elapsed/(i+1):.2f}s/row, v_tokens={visual_token_num}")

df["prediction"] = predictions
df.to_csv("chartqa_predictions_llavanext_cdpruner.csv", index=False)
print("Saved chartqa_predictions_llavanext_cdpruner.csv")
