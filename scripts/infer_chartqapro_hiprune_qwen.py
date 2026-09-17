import os
import time
import ast
import pandas as pd
import torch

os.environ["HIPRUNE_QWEN_RETENTION"] = "0.223"  # compressed
os.environ["HIPRUNE_ALPHA"] = "0.1"
os.environ["HIPRUNE_OBJECT_LAYER"] = "16"

import sys
sys.path.insert(0, "/scratch/bh2863/fastv_project/HiPrune/Qwen2_5_VL")
from qwen2_5_vl_HiPrune import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def prompt_context(question, answer, q_type, vqa_type):
    assert vqa_type in ["Direct", "CoT", "PoT"]
    assert q_type in ["Factoid", "Multi Choice", "Conversational", "Fact Checking", "Hypothetical"]
    if vqa_type == "Direct":
        if q_type == "Factoid":
            question_context = f'''
            You are given a factoid question that you need to answer based on the provided image.
            Your answer should be a single word, number, or phrase. If the question is unanswerable based on
            the information in the provided image, your answer should be unanswerable. Do not generate units.
            But if numerical units such as million, m, billion, B, or K are required, use the exact notation
            shown in the chart.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Question: {question[0]}
            '''
        elif q_type == "Multi Choice":
            question_context = f'''
            You are given a question along with different possible answers. You need to select the correct answer
            from them based on the provided image.
            Your answer should be one of the options letters only: a, b, c or d (just the letter itself without any
            additional text). If the question is unanswerable based on the information in the provided image, your
            answer should be unanswerable.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Question: {question[0]}
            '''
        elif q_type == "Conversational":
            question_context = f'''
            You are given a multi-turn conversation, and your job is to answer the final question based on the
            conversation history and the information in the provided image.
            Your answer should be a single word, number, or phrase. If the question is unanswerable based on
            the information in the provided image, your answer should be unanswerable. Do not generate units.
            But if numerical units such as million, m, billion, B, or K are required, use the exact notation
            shown in the chart.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Conversation: {[x for qa in zip(question[:-1], answer[:-1]) for x in qa]} Question: {question[-1]}
            '''
        elif q_type == "Fact Checking":
            question_context = f'''
            You are given a fact statement that you need to assess based on the provided image.
            Your answer should be either true or false (without any additional text). If the question is
            unanswerable based on the information in the provided image, your answer should be unanswerable.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Question: {question[0]}
            '''
        elif q_type == "Hypothetical":
            question_context = f'''
            You are given a hypothetical question that you need to answer based on the provided image.
            Your answer should be a single word, number, or phrase. If the question is unanswerable based on
            the information in the provided image, your answer should be unanswerable. Do not generate units.
            But if numerical units such as million, m, billion, B, or K are required, use the exact notation
            shown in the chart.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Question: {question[0]}
            '''
    return question_context


model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model = model.to("cuda")
processor = AutoProcessor.from_pretrained(
    model_path,
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28,
)

df = pd.read_csv("chartqapro_plain.csv")
print(f"Loaded {len(df)} rows")

predictions = []
start = time.time()

for i, row in df.iterrows():
    question = ast.literal_eval(row["question"])
    answer = ast.literal_eval(row["answer"])
    question_type = row["question_type"]
    paragraph = row["paragraph"]
    if pd.isna(paragraph):
        paragraph = ""

    question_context = prompt_context(question, answer, question_type, "Direct")
    prompt_text = (paragraph + "\n" + question_context) if paragraph else question_context

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": row["image_path"]},
            {"type": "text", "text": prompt_text},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=200)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    predictions.append(pred)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"{i+1}/{len(df)} done, {elapsed:.1f}s elapsed, {elapsed/(i+1):.2f}s/row")

df["prediction"] = predictions
df.to_csv("chartqapro_predictions_hiprune_qwen.csv", index=False)
print("Saved chartqapro_predictions_hiprune_qwen.csv")
