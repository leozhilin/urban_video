# encoding = utf-8

import time
import json
import asyncio

import pandas as pd
from src.clients.qwen_client import qwen_client
from src.agent_prompt import planner_agent_prompt
from src.function_call import FunctionCall
from src.tools.video_summary import video_summary_tool

video_agent_tools = [
    video_summary_tool
]

async def video_agent_loop(video_path, question, question_category):
    # 执行对话循环
    response_text = await execute_conversation_loop(
        qwen_client,
        video_path, 
        question, 
        question_category
    )
    return response_text

async def execute_conversation_loop(llm_client, video_path, question, question_category):
    """执行对话循环"""
    initial_messages = [
        {"role": "system", "content": planner_agent_prompt},
        {"role": "user", "content": f"视频路径: {video_path}, 问题: {question}, 问题类别: {question_category}"}
    ]
    print(f"initial_messages: {initial_messages}\n\n")

    turn_messages = initial_messages

    while True:
        response = await call_model(llm_client, turn_messages)

        response_text, response_function_calls = extract_text_and_function_call_infos(response)
        print("response_text: ", response_text)
        print("response_function_calls: ", response_function_calls)

        if response_function_calls:
            # 有工具调用的处理流程
            tool_call_request = {
                "role": "assistant",
                "content": response_text,
                "tool_calls": [{
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"]
                    }
                } for call in response_function_calls]
            }
            turn_messages.append(tool_call_request)

            tool_call_messages = await handle_function_calls(
                response_text,
                response_function_calls,
            )
            turn_messages.extend(tool_call_messages)
            print(f"tool_call_messages: {tool_call_messages}\n\n")
        else:
            # 没有工具调用的处理流程
            response_message = {
                "role": "assistant",
                "content": response_text
            }
            turn_messages.append(response_message)
            print(f"final response_message: {response_message}\n\n")
            return response_text
            break

async def call_model(qwen_client, messages):
    """调用模型"""
    response = qwen_client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=video_agent_tools,
        tool_choice="auto"
    )
    return response

def extract_text_and_function_call_infos(response):
    """提取函数调用和对应的文本描述信息"""
    response_function_calls = []
    response_text = response.choices[0].message.content

    # 检查是否有tool_calls
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            # 将tool_call转换为字典格式
            tool_call_dict = {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
                "id": tool_call.id,
                "type": tool_call.type
            }
            response_function_calls.append(tool_call_dict)
    
    # 检查是否有function_call
    elif hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call:
        function_call = response.choices[0].message.function_call
        function_call_dict = {
            "name": function_call.name,
            "arguments": function_call.arguments
        }
        response_function_calls.append(function_call_dict)

    return response_text, response_function_calls


async def handle_function_calls(
    response_text,
    response_function_calls,
):
    """处理函数调用"""
    print(f"Function calls: {response_function_calls}\n\n")

    if not response_function_calls:
        return []

    function_handler = FunctionCall()

    # 目前不存在同时执行多个function的情况
    function_call_info = response_function_calls[0]

    tool_call_result_message = await function_handler.handle_tool_call(function_call_info)

    return [tool_call_result_message]


if __name__ == "__main__":
    import asyncio
    import uuid

    # 读取数据集
    df = pd.read_parquet('UrbanVideo-Bench/MCQ.parquet')
    # 随机打乱数据
    df = df.sample(frac=1, random_state=40)
    # 只取前20个样本
    df = df.head(20)
    
    print(f"Processing {len(df)} randomly selected samples")

    for index, row in df.iterrows():
        video_id = row['video_id']
        question = row['question']
        answer = row['answer']
        question_category = row['question_category']
        video_path = f"/home/liuzhilin/myproj/UrbanVideo-Bench/videos/{video_id}"

        agent_response_text = asyncio.run(video_agent_loop(
            video_path,
            question, 
            question_category
        ))
        result = {
            "video_id": video_id,
            "question": question,
            "answer": answer,
            "question_category": question_category,
            "agent_response_text": agent_response_text
        }
        file_name = f"results.json"
        with open(file_name, "a") as f:
            json.dump(result, f)
            f.write("\n")
        print("ground truth answer: ", answer)
        print("--------------------------------")