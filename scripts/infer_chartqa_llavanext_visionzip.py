import os, time, pandas as pd, torch
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from visionzip import visionzip

model_path = "liuhaotian/llava-v1.6-vicuna-7b"
conv_mode = "llava_v1"

print("Loading model...")
disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

model = visionzip(model, dominant=108, contextual=20)   # 642 = 22.3% of 2880
print("visionzip:", model.model.vision_tower.vision_tower._info)

df = pd.read_csv("chartqa_plain.csv")
print(f"Loaded {len(df)} rows")

predictions, start = [], time.time()
for i, row in df.iterrows():
    image = Image.open(row["image_path"]).convert("RGB")
    qs = DEFAULT_IMAGE_TOKEN + "\n" + str(row["question"]) + " Answer the question using a single word or phrase."
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=False, max_new_tokens=200, use_cache=True,
        )
    output_ids = out[0] if isinstance(out, tuple) else out
    predictions.append(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())

    if (i + 1) % 50 == 0:
        el = time.time() - start
        print(f"{i+1}/{len(df)} done, {el:.1f}s, {el/(i+1):.2f}s/row")

df["prediction"] = predictions
df.to_csv("chartqa_predictions_llavanext_visionzip.csv", index=False)
print("Saved chartqa_predictions_llavanext_visionzip.csv")
