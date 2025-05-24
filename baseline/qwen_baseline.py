import cv2
import base64
import time
from openai import OpenAI
import os
import pandas as pd
import warnings
import math
import copy
import json

warnings.filterwarnings("ignore")


# 主执行块
if __name__ == '__main__':
    model = 'qwen-vl-max-latest'
    client = OpenAI(
        api_key="sk-b4e6ca86b23843cfa9e138afdd832989",  # 替换为你的OpenAI API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 替换为你的OpenAI API基础URL
    )

    # 数据集路径
    folder_path = 'UrbanVideo-Bench/videos'  # 定义存储视频文件的文件夹路径
    
    # 从JSONL文件读取数据
    data = []
    with open('baseline/test_samples.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    QA_df = pd.DataFrame(data)

    # 定义保存结果的文件夹路径，如果不存在则创建
    folder_path_result = 'baseline/qwen_result'
    if not os.path.exists(folder_path_result):
        os.makedirs(folder_path_result)

    # 定义结果CSV文件的路径
    res_path = os.path.join(folder_path_result, '%s_output.csv' % model)

    # 检查结果文件是否已存在
    if os.path.exists(res_path):
        # 如果文件存在，加载它并找到'Output'列中最后一个有效索引
        res = pd.read_csv(res_path, index_col=0)
        last_valid_index = int(res['Output'].last_valid_index())
        last_valid_index += 1  # 从下一个索引开始处理
    else:
        # 如果文件不存在，基于QA数据集创建新的DataFrame
        res = copy.deepcopy(QA_df)
        res['Output'] = None  # 添加初始化为None的'Output'列
        last_valid_index = 0  # 从第一个索引开始处理

    # 从最后一个有效索引开始遍历每个问题
    for qa_idx in range(last_valid_index, res.shape[0]):
        print('正在处理索引: %d' % qa_idx)

        # 获取当前问题的视频ID
        select_vid_name = res['video_id'].iloc[qa_idx]

        # 使用OpenCV打开视频文件
        video = cv2.VideoCapture(os.path.join(folder_path, str(select_vid_name)))
        video_fps = video.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率(FPS)

        # 初始化存储base64编码帧的列表
        base64Frames = []
        while video.isOpened():
            success, frame = video.read()  # 从视频中读取一帧
            if not success:
                break  # 如果没有更多帧则停止读取

            # 将帧编码为JPEG图像并转换为base64格式
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        # 释放视频文件并打印读取的帧数
        video.release()
        print(len(base64Frames), "frames read.")

        # 创建提示词，让GPT模型基于视频回答问题
        prompt = "This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions: \n"
        prompt += "The template for the answer is: \n\
                        Option: []; Reason: []\n\
                        where the Option only outputs one option from 'A' to 'E' here, do not output redundant content. Reason explains why you choose this option."

        # 将数据集中的问题添加到提示词中
        qa = res['question'].iloc[qa_idx]
        prompt += '\n' + qa

        try:
            # 选择帧的子集以减少发送给模型的帧数
            div_num = math.ceil(len(base64Frames) / 32)  # 将帧分成块  最多会送32帧到VLM
            base64Frames_selected = base64Frames[0::div_num]  # 选择每隔n帧

            # 为GPT模型准备base64格式的视频内容
            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            for buffer in base64Frames_selected:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{buffer}"
                    }})

            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            # 将提示词发送给GPT模型并获取响应
            result = client.chat.completions.create(
                model=model,
                messages=PROMPT_MESSAGES
            )
            print(result.choices[0].message.content)  # 打印模型的响应
            res_str = result.choices[0].message.content  # 提取响应字符串

            # 将响应保存在结果DataFrame的'Output'列中
            res['Output'].iloc[qa_idx] = res_str

        except Exception as e:
            # 处理错误并在重试前等待60秒
            print(f"发生错误: {e}")
            time.sleep(60)

        # 将更新后的结果DataFrame保存到CSV文件
        res.to_csv(res_path)