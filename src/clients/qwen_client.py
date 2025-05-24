import os
from openai import OpenAI

qwen_client = OpenAI(
    api_key="sk-b4e6ca86b23843cfa9e138afdd832989",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
if __name__ == "__main__":
    completion = qwen_client.chat.completions.create(
        model="qwen-plus", 
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}
            ]
    )
    print(completion.choices[0].message.content)