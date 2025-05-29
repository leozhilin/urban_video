import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
import json
import copy

# Set model path
model_path = "Video-R1/Video-R1-7B"

# Load dataset
data = []
with open('baseline/test_samples.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
QA_df = pd.DataFrame(data)

# Define result folder path
folder_path_result = 'baseline/video_r1_result'
if not os.path.exists(folder_path_result):
    os.makedirs(folder_path_result)

# Define result CSV path
res_path = os.path.join(folder_path_result, 'video_r1_output.csv')

# Check if result file exists
if os.path.exists(res_path):
    res = pd.read_csv(res_path, index_col=0)
    last_valid_index = int(res['Output'].last_valid_index())
    last_valid_index += 1
else:
    res = copy.deepcopy(QA_df)
    res['Output'] = None
    last_valid_index = 0

# Video folder path
video_folder_path = 'UrbanVideo-Bench/videos'

# Question type 
problem_type = 'multiple choice'

# Prompt template
QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

# Question type 
TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
}

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="./models",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = "left" 

processor.tokenizer = tokenizer

# Process each question
for qa_idx in range(last_valid_index, res.shape[0]):
    try:
        print('Processing index: %d' % qa_idx)
        
        # Get current video ID and question
        video_path = os.path.join(video_folder_path, str(res['video_id'].iloc[qa_idx]))
        question = res['question'].iloc[qa_idx]

        # Construct multimodal message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 200704,
                        "nframes": 32
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[problem_type]
                    },
                ],
            }
        ]

        # Convert to prompt string
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process video input
        image_inputs, video_inputs = process_vision_info(messages)

        # Ensure video inputs are in correct format
        if isinstance(video_inputs, list):
            video_inputs = video_inputs[0]

        # Prepare model inputs
        inputs = processor(
            text=prompt,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Save output using loc to avoid chained assignment
        res.loc[qa_idx, 'Output'] = output_text[0]
        res.to_csv(res_path)
        print(output_text[0])
    except Exception as e:
        pass
