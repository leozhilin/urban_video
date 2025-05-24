MODEL_NAME = "qwen-vl-max-2025-04-02"


import asyncio
import traceback
import pandas as pd
import os
from dashscope import MultiModalConversation
from tqdm import tqdm

def chat_with_qwen(video_path, question):
    messages = [
        {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
        {'role':'user', 'content': [
            {'image': f"file://{video_path}"},
            {'text': question}
        ]}
    ]

    response = MultiModalConversation.call(
        api_key="sk-b4e6ca86b23843cfa9e138afdd832989",
        model='qwen-vl-max-latest', 
        messages=messages
    )
    print(response)

async def main():
    # 读取数据集
    df = pd.read_parquet('UrbanVideo-Bench/MCQ.parquet') 
    # 随机打乱数据
    df = df.sample(frac=1, random_state=42)
    # 只取前20个样本
    df = df.head(20)
    print(f"Processing {len(df)} randomly selected samples")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            video_id = row['video_id']
            question = row['question']
            answer = row['answer']
            question_category = row['question_category']
            video_path = f"/home/liuzhilin/myproj/UrbanVideo-Bench/videos/{video_id}"
            chat_with_qwen(video_path, question)
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error processing video {video_id}: {e}")

if __name__ == "__main__":
    asyncio.run(main())


