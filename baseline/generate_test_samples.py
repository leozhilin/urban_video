import pandas as pd
import json

def generate_samples():
    # Read the dataset
    df = pd.read_parquet('UrbanVideo-Bench/MCQ.parquet')
    
    # Randomly select 100 samples
    df = df.sample(n=4000, random_state=42)
    
    # Save as JSONL file
    with open('baseline/test_samples.jsonl', 'w') as f:
        for _, row in df.iterrows():
            # Convert row to dict and write as JSON line
            sample = {
                'video_id': row['video_id'],
                'question': row['question'],
                'answer': row['answer'],
                'question_category': row['question_category'],
                'Question_id': row['Question_id']
            }
            f.write(json.dumps(sample) + '\n')
    
    print(f"Successfully saved 100 random samples to baseline/test_samples.jsonl")

if __name__ == "__main__":
    generate_samples() 