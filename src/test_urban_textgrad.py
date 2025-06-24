import os
import re
import cv2
import base64
import json
import pandas as pd
from src.textgrad_urban import UrbanTextGrad
import math

def truncate_text(text, max_length=100):
    """截断文本到指定长度，并添加省略号"""
    if not isinstance(text, str):
        return str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def print_header(text):
    """打印带装饰的标题"""
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_step(step_name, details=""):
    """打印处理步骤"""
    print(f"🔄 {step_name}")
    if details:
        print(f"   {details}")

def print_success(message):
    """打印成功消息"""
    print(f"✅ {message}")

def print_error(message):
    """打印错误消息"""
    print(f"❌ {message}")

def print_info(message):
    """打印信息消息"""
    print(f"ℹ️  {message}")

def read_video_frames(video_path: str, max_frames: int = 32) -> list:
    """Read video frames and convert to base64."""
    video = cv2.VideoCapture(video_path)
    frames = []
    
    # 首先读取所有帧
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        _, buffer = cv2.imencode(".jpg", frame)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    video.release()
    
    # 如果帧数超过max_frames，进行均匀采样
    if len(frames) > max_frames:
        div_num = math.ceil(len(frames) / max_frames)  # 计算采样间隔
        frames = frames[0::div_num]  # 均匀采样
    
    return frames

def extract_option_letter(text):
    """Extract option letter from text."""
    if not isinstance(text, str):
        return None
    
    # Try to find option in the format "Option: X"
    match = re.search(r'Option:\s*[\[\s]*(\w)', text)
    if match:
        return match.group(1)
    
    # Try to find option in the format "Final Answer: X"
    match = re.search(r'Final Answer:\s*[\[\s]*(\w)', text)
    if match:
        return match.group(1)
    
    # If no match found, try to get the first letter
    return text[0].upper() if text else None

def main():
    print_header("Urban TextGrad 测试开始")
    
    # Initialize TextGrad with your model
    print_step("初始化 TextGrad 模型")
    # textgrad = UrbanTextGrad(model_name="qwen-vl-max-latest")
    textgrad = UrbanTextGrad()
    print_success("TextGrad 模型初始化完成")
    
    # Load test samples
    print_step("加载测试样本")
    with open('baseline/test_samples.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    QA_df = pd.DataFrame(data)
    print_success(f"成功加载测试样本")
    
    # Create results directory if it doesn't exist
    results_dir = 'textgrad_results'
    os.makedirs(results_dir, exist_ok=True)
    print_info(f"结果将保存到: {results_dir}")
    
    print_header("开始处理测试样本")
    
    # Process each sample
    for idx, row in QA_df.iterrows(): 
        # if row['question_category'] == "Action Generation":
        if  idx >= 0:
        # if idx >= 714:
            try:
                print(f"\n{'='*50}")
                print(f"📊 处理样本 {idx+1} (ID: {idx})")
                print(f"📹 视频ID: {row['video_id']}")
                print(f"📝 问题类别: {row['question_category']}")
                print(f"❓ 问题: {truncate_text(row['question'], 80)}")
                print(f"✔️  正确答案: {row['answer']}")
                print(f"{'='*50}")
                
                # Read video frames
                print_step("读取视频帧")
                video_path = os.path.join('UrbanVideo-Bench/videos', str(row['video_id']))
                print_info(f"视频路径: {video_path}")
                
                frames = read_video_frames(video_path)
                print_success(f"成功读取 {len(frames)} 帧")
                
                # Process with TextGrad
                print_step("使用 TextGrad 处理")
                print_info("正在生成初始答案...")
                
                result = textgrad.process(
                    question=row['question'],
                    frames=frames,
                    video_path=video_path,
                    question_category=row['question_category']
                )
                
                print_success("TextGrad 处理完成")
                
                # Extract and display results
                final_answer = result["final_answer"]
                init_answer = result["init_answer"]
                init_option = extract_option_letter(init_answer)
                final_option = extract_option_letter(final_answer)
                
                print("\n📋 处理结果:")
                print(f"   初始答案: {truncate_text(init_answer)}")
                print(f"   初始选项: {init_option}")
                print(f"   最终答案: {truncate_text(final_answer)}")
                print(f"   最终选项: {final_option}")
                print(f"   正确答案: {row['answer']}")
                
                # Check if initial answer is correct
                if init_option == row['answer']:
                    print_success("初始答案正确！保存结果...")
                    
                    # Save result
                    output_path = os.path.join(results_dir, f'sample_{idx}.json')
                    with open(output_path, 'w') as f:
                        json.dump({
                            'video_id': row['video_id'],
                            'question': row['question'],
                            'question_category': row['question_category'],
                            'ground_truth': row['answer'],
                            'textgrad_result': result
                        }, f, indent=2)
                    
                    print_success(f"结果已保存: {output_path}")
                else:
                    print_info("初始答案不正确, 保存结果...")
                
            except Exception as e:
                print_error(f"处理样本 {idx+1}/{len(QA_df)} 时发生错误:")
                print_error(f"错误详情: {str(e)}")
                print_info("继续处理下一个样本...")
    
    print_header("测试完成")
    print_success("所有样本处理完毕")

if __name__ == "__main__":
    main() 