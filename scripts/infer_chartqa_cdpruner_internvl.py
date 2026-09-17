import os, sys, time
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

os.environ["CDPRUNER_RETAIN_RATIO"] = "0.223"

from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

weight_path = "/scratch/bh2863/fastv_project/InternVL3-8B-cdpruner"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image_tiles(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values, tiles

print("Loading InternVL model...")
model = AutoModel.from_pretrained(weight_path, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(weight_path, trust_remote_code=True, use_fast=False)
model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def compute_tile_relevance(tile_img, question, grid_size=16):
    inputs = clip_processor(text=[question], images=tile_img, return_tensors="pt",
                             padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        patch_embeds = vision_out.last_hidden_state[:, 1:, :]
        patch_embeds = clip_model.visual_projection(patch_embeds)
        text_embeds = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        patch_embeds = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        sim = patch_embeds[0] @ text_embeds[0]
        side = int(sim.shape[0] ** 0.5)
        sim_grid = sim.view(1, 1, side, side)
        sim_resized = F.interpolate(sim_grid, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
        return sim_resized.flatten().to(torch.bfloat16)

df = pd.read_csv("chartqa_plain.csv")
print(f"Loaded {len(df)} rows")

predictions = []
start = time.time()

for i, row in df.iterrows():
    pixel_values, tiles = load_image_tiles(row["image_path"], input_size=448, max_num=6)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    clip_relevance = torch.cat([
        compute_tile_relevance(t, str(row["question"]), grid_size=16) for t in tiles
    ])

    question = str(row["question"]) + " Answer the question using a single word or phrase."
    generation_config = dict(max_new_tokens=200, do_sample=False, clip_relevance=clip_relevance)

    response = model.chat(tokenizer, pixel_values, question, generation_config)
    predictions.append(response.strip())

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"{i+1}/{len(df)} done, {elapsed:.1f}s elapsed, {elapsed/(i+1):.2f}s/row")

df["prediction"] = predictions
df.to_csv("chartqa_predictions_cdpruner_internvl.csv", index=False)
print("Saved chartqa_predictions_cdpruner_internvl.csv")
