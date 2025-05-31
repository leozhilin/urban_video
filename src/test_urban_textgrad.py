import os
import cv2
import base64
import json
import pandas as pd
from src.textgrad_urban import UrbanTextGrad

def read_video_frames(video_path: str, max_frames: int = 32) -> list:
    """Read video frames and convert to base64."""
    video = cv2.VideoCapture(video_path)
    frames = []
    
    while video.isOpened() and len(frames) < max_frames:
        success, frame = video.read()
        if not success:
            break
            
        _, buffer = cv2.imencode(".jpg", frame)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    video.release()
    return frames

def main():
    # Initialize TextGrad with your model
    # textgrad = UrbanTextGrad(model_name="qwen-vl-max-latest")
    textgrad = UrbanTextGrad()
    
    # Load test samples
    with open('baseline/test_samples.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    QA_df = pd.DataFrame(data)
    
    # Create results directory if it doesn't exist
    results_dir = 'textgrad_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each sample
    for idx, row in QA_df.iterrows():
        print(f"Processing sample {idx + 1}/{len(QA_df)}")
        
        # Read video frames
        video_path = os.path.join('UrbanVideo-Bench/videos', str(row['video_id']))
        frames = read_video_frames(video_path)
        
        # Process with TextGrad
        result = textgrad.process(
            question=row['question'],
            frames=frames,
        )
        
        # Save result
        output_path = os.path.join(results_dir, f'sample_{idx}.json')
        with open(output_path, 'w') as f:
            json.dump({
                'video_id': row['video_id'],
                'question': row['question'],
                'ground_truth': row['answer'],
                'textgrad_result': result
            }, f, indent=2)
        
        print(f"Saved result to {output_path}")

if __name__ == "__main__":
    main() 