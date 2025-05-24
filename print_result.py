import json

def print_results():
    """
    Print the answer and agent_response_text fields from results.json
    """
    try:
        # Read the results.json file
        with open('results.json', 'r', encoding='utf-8') as f:
            # Parse each line as a separate JSON object
            for line in f:
                try:
                    data = json.loads(line.strip())
                    print("\nVideo ID:", data.get('video_id'))
                    print("Question:", data.get('question'))
                    print("Answer:", data.get('answer'))
                    print("Agent Response:", data.get('agent_response_text'))
                    print("-" * 80)  # Separator line
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print("Error: results.json file not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print_results()