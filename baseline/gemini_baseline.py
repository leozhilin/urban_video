import traceback
import asyncio
import os
from src.clients._gemini_api import AsyncGeminiVideoClient
from tqdm import tqdm
import json
from tenacity import retry, stop_after_attempt, wait_exponential

os.environ["GEMINI_API_KEY"] = "AIzaSyBcT5KUgqSambiGIz-KuW5Td4MJIx5ld8w"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def process_video(gemini_client, video_id, video_path, question, answer, question_category):
    try:
        # 读取视频文件
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        # 上传视频到 Gemini
        uri = await gemini_client.upload_to_gemini(
            file_data=video_data,
            mime_type="video/mp4",
            display_name=video_id
        )
        print(f"Video {video_id} uploaded successfully. URI: {uri}")

        # 等待文件激活
        await gemini_client.wait_for_active_status(uri)
        print(f"Video {video_id} is now active")

        # 准备调用模型
        model_name = "gemini-2.0-flash"
        history = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Please assume the role of an aerial agent.\n" \
                               "The video represents your egocentric observations from the past to the present.\n" \
                               "Please answer the following questions:\n\n" \
                               f"Question: {question}\n\n" 
                    },
                    {
                        "fileData": {
                            "mimeType": "video/mp4",
                            "fileUri": uri,
                        }
                    },
                    {
                        "text": "The template for the answer is:\n" \
                               "Option: [] (Only output one option from 'A' to 'E' here, do not output redundant content)\n" \
                               "Reason: [] (Explain why you choose this option)"
                    }
                ]
            }
        ]
        
        generation_config = {
            "temperature": 0,
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
        
        # 删除视频
        await gemini_client.delete_video(uri)
        
        return response
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        print(traceback.format_exc())
        # 确保在出错时也尝试删除视频
        try:
            if 'uri' in locals():
                await gemini_client.delete_video(uri)
        except:
            pass
        return None

async def main():
    # 读取JSONL文件
    samples = []
    with open('baseline/test_samples.jsonl', 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Processing {len(samples)} samples from baseline/test_samples.jsonl")
    
    # 初始化 Gemini 客户端
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("请设置 GEMINI_API_KEY 环境变量")
        return
    
    gemini_client = AsyncGeminiVideoClient(api_key=api_key)
    
    try:
        # 处理每个视频的所有问题
        results = {}
        for sample in tqdm(samples):
            try:
                video_id = sample['video_id']
                question = sample['question']
                answer = sample['answer']
                question_category = sample['question_category']
                
                video_path = f"UrbanVideo-Bench/videos/{video_id}"
                if os.path.exists(video_path):
                    result = await process_video(
                        gemini_client, 
                        video_id, 
                        video_path, 
                        question, 
                        answer, 
                        question_category
                    )
                    # 使用问题ID作为键来存储结果
                    results[sample['Question_id']] = {
                        'video_id': video_id,
                        'question': question,
                        'answer': answer,
                        'question_category': question_category,
                        'gemini_response': result
                    }
                    # 每处理完一个样本就保存一次结果
                    with open('baseline/gemini_analysis_results.json', 'w') as f:
                        json.dump(results, f, indent=2)
                else:
                    print(f"Video file not found: {video_path}")
            except Exception as e:
                print(f"Error processing sample {sample['Question_id']}: {str(e)}")
                continue
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current results...")
    finally:
        # 保存最终结果
        with open('gemini_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        await gemini_client.close()

if __name__ == "__main__":
    asyncio.run(main())