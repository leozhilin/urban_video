import os
import pandas as pd
import re
import json
from glob import glob

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

def process_textgrad_results():
    # Path to the results directory
    results_dir = 'textgrad_results'
    
    # Get all json files in the directory
    json_files = glob(os.path.join(results_dir, 'sample_*.json'))
    
    # Initialize lists to store data
    data = []
    
    # Process each json file
    for json_file in json_files:
        with open(json_file, 'r') as f:
            result = json.load(f)
            
            # Extract the final answer from textgrad_result
            final_answer = result['textgrad_result']['final_answer']
            extracted_option = extract_option_letter(final_answer)
            
            # Create a row of data
            row = {
                'video_id': result['video_id'],
                'question': result['question'],
                'ground_truth': result['ground_truth'],
                'model_answer': final_answer,
                'extracted_option': extracted_option
            }
            data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate accuracy
    df['is_correct'] = df['extracted_option'] == df['ground_truth']
    
    # Group by question category (you may need to extract this from the question text)
    # For now, we'll just calculate overall accuracy
    total_correct = df['is_correct'].sum()
    total_accuracy = total_correct / len(df)
    
    # Create accuracy summary
    accuracy_data = [{
        'Category': 'Total',
        'Accuracy': total_accuracy,
        'Num': len(df)
    }]
    
    # Convert to DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)
    
    # Save results
    output_dir = 'textgrad_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Save accuracy results
    accuracy_df.to_excel(os.path.join(output_dir, 'accuracy_results.xlsx'), index=False)
    
    print(f"Processed {len(json_files)} samples")
    print(f"Overall accuracy: {total_accuracy:.2%}")
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    process_textgrad_results() 