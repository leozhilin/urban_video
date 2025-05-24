# encoding: utf-8
# 创建项目工具
import asyncio
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential

from src.clients._gemini_api import AsyncGeminiVideoClient

video_summary_tool = {
    "type": "function",
    "function": {
        "name": "video_summary",
        "description": "这个工具会为围绕着用户问题为视频添加匹配内容的字幕，同时会按时间轴生成视频的摘要",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string", 
                    "description": "视频路径"
                },
                "question": {
                    "type": "string",
                    "description": "可以是用户的原始问题，也可以是你对问题加工修改后，更有利于取得你想要信息的问题，注意：你只需要提交问题，不要提交选项"
                }
            },
            "required": ["video_path", "question"]
        }
    }
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def process_video(gemini_client, video_path, question):
    # 读取视频文件
    with open(video_path, 'rb') as f:
        video_data = f.read()
    
    # 上传视频到 Gemini
    uri = await gemini_client.upload_to_gemini(
        file_data=video_data,
        mime_type="video/mp4",
        display_name=video_path
    )
    print(f"Video {video_path} uploaded successfully. URI: {uri}")

    # 等待文件激活
    await gemini_client.wait_for_active_status(uri)
    print(f"Video {video_path} is now active")

    # 准备调用模型
    model_name = "gemini-2.0-flash"
    history = [
        {
            "role": "user",
            "parts": [
                {
                    "text": "你是一位专业的视频分析师。你的任务是分析视频并生成详细字幕，重点关注如何回答给定的问题。\n\n" \
                            "请按照以下步骤进行：\n" \
                            "1. 仔细观看视频，识别关键场景、对象、动作及其关系。\n" \
                            "2. 为视频每5秒生成一条详细字幕。\n" \
                            "3. 重点关注与问题相关的内容。\n" \
                            "4. 包含对象、动作、对象间相对位置等的具体细节。\n\n" \
                            f"问题：{question}\n\n" \
                            "请按以下格式提供分析：\n" \
                            "1. 整体描述：[视频内容的简要总结]\n" \
                            "2. 详细字幕：\n" \
                            "   - 0-5秒：[描述这个时间段发生的事情]\n" \
                            "   - 5-10秒：[描述这个时间段发生的事情]\n" \
                            "   ...\n" \
                            "3. 关键点：[列出与问题相关的最重要观察]\n" \
                            "4. 时间轴摘要：[按时间顺序总结关键事件]"
                },
                {
                    "fileData": {
                        "mimeType": "video/mp4",
                        "fileUri": uri,
                    }
                }
            ]
        }
    ]
    
    generation_config = {
        "temperature": 0.2,  # 降低温度以获得更确定的输出
        "topP": 0.95,
        "topK": 40,
        "maxOutputTokens": 8192,
    }

    # 调用模型
    response = await gemini_client.call_model(
        model_name=model_name,
        system_instruction=None,
        history=history,
        generation_config=generation_config
    )
    response_text = response["candidates"][0]["content"]["parts"][0]["text"]
    print(response_text)
    
    # 删除视频
    await gemini_client.delete_video(uri)
    
    return {"response_text": response_text}

def parse_subtitles(subtitles_section):
    """解析字幕部分为结构化数据"""
    subtitles = []
    lines = subtitles_section.split("\n")
    
    for line in lines:
        if ":" in line:
            time_range, text = line.split(":", 1)
            time_range = time_range.strip().replace("-", "").replace("s", "")
            start_time, end_time = map(int, time_range.split())
            
            subtitles.append({
                "start_time": start_time,
                "end_time": end_time,
                "text": text.strip()
            })
    
    return subtitles

async def video_summary(video_path, question):
    """创建项目并初始化默认轨道"""
    try:
        gemini_client = AsyncGeminiVideoClient(api_key="AIzaSyBcT5KUgqSambiGIz-KuW5Td4MJIx5ld8w")
        result = await process_video(gemini_client, video_path, question)
        return result
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        print(traceback.format_exc())
        return {
            "output": "error",
            "error": str(e)
        }
    
if __name__ == "__main__":
    asyncio.run(video_summary("UrbanVideo-Bench/videos/AerialVLN_0_30BUDKLTXMTIKYKFIPPOXCMISXQE5S.mp4", "视频描述了哪些建筑"))
