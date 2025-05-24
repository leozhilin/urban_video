# encoding = utf-8
import aiohttp
import asyncio
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_fixed
import io
import traceback

class AsyncGeminiVideoClient:
    def __init__(self, api_key, base_url="https://generativelanguage.googleapis.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = aiohttp.ClientSession()

    async def close(self):
        """关闭 HTTP 会话和线程池。"""
        await self.session.close()
        # self.process_pool.shutdown()

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
    async def call_model(self, system_instruction, model_name, history, generation_config, tools = None, tool_mode = "AUTO"):
        """异步调用 Gemini 模型，带有重试机制。"""
        headers = {"Content-Type": "application/json"}
        
        if system_instruction:
            payload = {
                "systemInstruction": {
                    "role": "user",
                    "parts": [
                        {"text": system_instruction}
                    ]
                },
                "contents": history,
                "generationConfig": generation_config
            }
        else:
            payload = {
                "contents": history,
                "generationConfig": generation_config
            }

        # 函数调用
        if tools != None:
            payload["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": tool_mode
                }
            }
            payload["tools"] = [{"functionDeclarations": tools}]
            

        async with self.session.post(
            f"{self.base_url}/v1beta/models/{model_name}:generateContent?key={self.api_key}",
                headers=headers,
                json=payload,
            ) as response:
            response.raise_for_status()
            return await response.json()
        
    async def is_uri_active(self, file_uri):
        """轮询 API，直到文件状态为 'ACTIVE' 或 'FAILED'。"""
        get_url = f"{file_uri}?key={self.api_key}"
        async with self.session.get(get_url) as response:
            if response.status == 200:
                file_info = await response.json()
                # print(f"File info: {file_info}")
                state = file_info.get("state")
                if state == "ACTIVE":
                    # print(f"uri {file_uri} active")
                    return True
            return False

    
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(3))
    async def upload_to_gemini(self, file_data, mime_type, display_name):
        """将字节数据上传到 Gemini。"""
        # 将字节数据包装为新的缓冲区对象
        buffer = io.BytesIO(file_data)
        num_bytes = len(file_data)

        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(num_bytes),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        metadata = {"file": {"display_name": display_name}}

        # 启动上传会话
        async with self.session.post(
            f"{self.base_url}/upload/v1beta/files?key={self.api_key}",
            headers=headers,
            json=metadata,
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to start upload: {response.status} {await response.text()}")
            upload_url = response.headers.get("X-Goog-Upload-URL") or response.headers.get("X-Goog-Upload-Url")
            if not upload_url:
                raise Exception("Upload URL not found in the response headers.")

        # 上传文件数据
        headers = {
            "Content-Length": str(num_bytes),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }
        buffer.seek(0)
        async with self.session.put(upload_url, headers=headers, data=buffer) as upload_response:
            if upload_response.status != 200:
                raise Exception(f"Failed to upload: {upload_response.status} {await upload_response.text()}")

            file_info = await upload_response.json()
            return file_info["file"]["uri"]
    
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(3))
    async def delete_video(self, file_uri):
        """从 Gemini 删除文件。"""
        file_name = file_uri.split("/")[-1]
        async with self.session.delete(
            f"{self.base_url}/v1beta/files/{file_name}?key={self.api_key}"
        ) as response:
            if response.status == 200:
                return f"Video {file_name} deleted successfully."
            else:
                return f"Video {file_name} deleted failed."
                # raise Exception(f"Failed to delete video: {await response.text()}")
    
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(3))
    async def wait_for_active_status(self, file_uri):
        """轮询 API，直到文件状态为 'ACTIVE' 或 'FAILED'。"""
        get_url = f"{file_uri}?key={self.api_key}"
        while True:
            async with self.session.get(get_url) as response:
                if response.status == 200:
                    file_info = await response.json()
                    # print(f"File info: {file_info}")
                    state = file_info.get("state")
                    if state == "ACTIVE":
                        return
                    elif state == "FAILED":
                        raise Exception("File processing failed.")
                await asyncio.sleep(5)

    def _get_mime_type(self, file_type):
        """根据文件扩展名确定 MIME 类型。"""
        mime_types = {
            "mp4": "video/mp4",
            "mpeg": "video/mpeg",
            "mov": "video/mov",
            "avi": "video/avi",
            "flv": "video/x-flv",
            "mpg": "video/mpg",
            "webm": "video/webm",
            "wmv": "video/wmv",
            "3gp": "video/3gpp",
        }
        return mime_types.get(file_type, "application/octet-stream")

# 测试主函数
if __name__ == "__main__":
    import os
    import requests
    os.environ["GEMINI_API_KEY"] = "AIzaSyBcT5KUgqSambiGIz-KuW5Td4MJIx5ld8w"
    async def main():
        api_key = os.getenv('GEMINI_API_KEY')  # 请确保环境变量中设置了 GEMINI_API_KEY
        if not api_key:
            print("请设置 GEMINI_API_KEY 环境变量。")
            return
        gemini_video_client = AsyncGeminiVideoClient(api_key=api_key)

        video_url = "https://aivlog-1313805896.cos.na-siliconvalley.myqcloud.com/uid_996a3bee-b1b2-4bbf-aab3-3b3470545f06/uid_ios_test/1729619094_0_3604.mp4"  # 替换为实际的视频 URL
        # 下载视频并生成byte
        video_data = requests.get(video_url).content
        
        try:
            # 测试上传视频
            uri = await gemini_video_client.upload_to_gemini(
                file_data=video_data,
                mime_type="video/mp4",
                display_name="test_video.mp4"
            )
            print(f"Video uploaded successfully. URI: {uri}")

            # # 测试等待文件激活
            file_info = await gemini_video_client.wait_for_active_status(uri)
            print(f"File info: {file_info}")
           
            print("All files are active.")

            # 测试调用模型
            system_instruction = "generate content for a vlog"
            model_name = "gemini-2.0-flash"
            # model_name = "gemini-1.5-pro"
            # model_name = "gemini-2.0-flash-exp"
            history = [
                {
                    "role":"user",
                    "parts":[
                        {
                            "text":"Video_id: 1734950355_3, duration 6 seconds, video shooting time: 2024-12-11 14:15:02\n"
                        },
                    ]
                },
                {
                    "role":"user",
                    "parts":[
                        {
                            "fileData":{
                                "mimeType":"video/mp4",
                                "fileUri":uri,
                            }
                        }
                    ]
                },
                {
                    "role":"user",
                    "parts":[
                        {
                            "text":"start_time & end_time: 00:00.00 - 00:00.60 shot_type:usable_non_highlight_shot, sharpness: clear\nvideo clip description: People holding sky lanterns\nstart_time & end_time: 00:00.60 - 00:06.00 shot_type:highlights, sharpness: clear\nvideo clip description: Sky lanterns and fireworks show\n"
                        }
                    ]
                }
            ]
            print(history)
            generation_config = {
                "temperature": 0,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 8192,
                "responseMimeType": "text/plain",
            }
            response = await gemini_video_client.call_model(
                system_instruction=system_instruction,
                model_name=model_name,
                history=history,
                generation_config=generation_config
            )
            print(f"Model Response: {response}")

        except Exception as e:
            traceback.print_exc()
            print(f"An error occurred: {e}")
        finally:
            await gemini_video_client.close()

    asyncio.run(main())
