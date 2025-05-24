import json
import re

def analyze_results(file_path='gemini_analysis_results.json'):
    # 读取结果文件
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # 初始化计数器
    total_questions = 0
    correct_answers = 0
    
    # 遍历所有问题
    for question_id, data in results.items():
        # 跳过没有响应的题目
        if data['gemini_response'] is None:
            continue
            
        total_questions += 1
        correct_answer = data['answer']
        
        # 从模型响应中提取答案
        try:
            model_response = data['gemini_response']['candidates'][0]['content']['parts'][0]['text']
            # 使用正则表达式提取选项
            option_match = re.search(r'Option:\s*\[?([A-E])\]?', model_response)
            if option_match:
                model_answer = option_match.group(1)
                # 比较答案
                if model_answer == correct_answer:
                    correct_answers += 1
            else:
                print(f"Warning: Could not extract option from response for question {question_id}")
                print(f"Response: {model_response}")
        except Exception as e:
            print(f"Error processing question {question_id}: {str(e)}")
            continue
    
    # 计算正确率
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    # 打印结果
    print(f"总题目数: {total_questions}")
    print(f"正确题目数: {correct_answers}")
    print(f"正确率: {accuracy:.2f}%")
    
    # 按问题类别统计
    category_stats = {}
    for question_id, data in results.items():
        if data['gemini_response'] is None:
            continue
            
        category = data['question_category']
        if category not in category_stats:
            category_stats[category] = {'total': 0, 'correct': 0}
            
        category_stats[category]['total'] += 1
        try:
            model_response = data['gemini_response']['candidates'][0]['content']['parts'][0]['text']
            option_match = re.search(r'Option:\s*\[?([A-E])\]?', model_response)
            if option_match:
                model_answer = option_match.group(1)
                if model_answer == data['answer']:
                    category_stats[category]['correct'] += 1
        except:
            pass
    
    # 打印每个类别的统计
    print("\n按问题类别统计:")
    for category, stats in category_stats.items():
        category_accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"\n{category}:")
        print(f"  题目数: {stats['total']}")
        print(f"  正确数: {stats['correct']}")
        print(f"  正确率: {category_accuracy:.2f}%")
        
    # 打印错误案例
    print("\n错误案例:")
    for question_id, data in results.items():
        if data['gemini_response'] is None:
            continue
            
        try:
            model_response = data['gemini_response']['candidates'][0]['content']['parts'][0]['text']
            option_match = re.search(r'Option:\s*\[?([A-E])\]?', model_response)
            if option_match:
                model_answer = option_match.group(1)
                if model_answer != data['answer']:
                    print(f"\n问题ID: {question_id}")
                    print(f"问题类别: {data['question_category']}")
                    print(f"问题: {data['question']}")
                    print(f"正确答案: {data['answer']}")
                    print(f"模型答案: {model_answer}")
                    print(f"模型解释: {model_response}")
        except:
            pass

if __name__ == "__main__":
    analyze_results()