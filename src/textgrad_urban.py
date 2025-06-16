import re
from typing import List, Dict, Any
from prompts import (
    REASONER_PROMPT, 
    EVALUATOR_PROMPT, 
    FEEDBACKER_PROMPT, 
    OPTIMIZER_PROMPT,
)
from src.clients.qwen_client import qwen_client
from src.clients._gemini_api import GeminiVideoClient
GEMINI_API_KEY = ["AIzaSyBcT5KUgqSambiGIz-KuW5Td4MJIx5ld8w", "AIzaSyANKqoVmbi4QKpLvi3g7aYHYNhuq2AByWs"] # 公司、个人gmail

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

class QwenModel():
    def __init__(self, model_name: str = "qwen-vl-max-latest"):
        self.model_name = model_name
        self.client = qwen_client
        
    def __call__(self, messages, system_prompt=None):
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=messages
        )
        return response.choices[0].message.content 

class GeminiModel():
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-05-20"):
        self.model_name = model_name
        self.client = GeminiVideoClient(api_key=GEMINI_API_KEY[1])

    def __call__(self, question, answer, video_path):
        video_file = None
        try:
            video_file = self.client.upload_file(video_path, mime_type="video/mp4")
            
            contents = [
                EVALUATOR_PROMPT.format(question=question, answer=answer),
                video_file,
            ]

            generation_config = {
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }

            # 调用模型
            response = self.client.call_model(
                model_name=self.model_name,
                contents=contents,
                generation_config=generation_config
            )
            # print("---- gemini response: ", response.text)
            return response.text
        finally:
            if video_file:
                self.client.delete_file(video_file)

class UrbanReasoner:
    def __init__(self, model: QwenModel):
        self.model = model
        
    def __call__(self, question: str, frames: List[str]) -> str:
        content = [{"type": "text", "text": REASONER_PROMPT.format(question=question)}]
        print("len of frames: ", len(frames))
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })
            
        messages = [{"role": "user", "content": content}]
        response = self.model(messages)
        return response

class UrbanEvaluator:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, question: str, answer: str, frames: List[str], video_path: str) -> str:
        if isinstance(self.model, QwenModel):
            content = [{"type": "text", "text": EVALUATOR_PROMPT.format(
                question=question,
                answer=answer
            )}]
            for frame in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                })
            messages = [{"role": "user", "content": content}]
            evaluation = self.model(messages)
        elif isinstance(self.model, GeminiModel):
            evaluation = self.model(question, answer, video_path)

        return evaluation

class UrbanFeedbacker:
    def __init__(self, model: QwenModel):
        self.model = model
        
    def __call__(self, question: str, answer: str, evaluation: str) -> str:
        messages = [{"role": "user", "content": FEEDBACKER_PROMPT.format(
            question=question,
            answer=answer,
            evaluation=evaluation
        )}]
        feedback = self.model(messages)
        return feedback

class UrbanOptimizer:
    def __init__(self, model: QwenModel):
        self.model = model
        
    def __call__(self, question: str, answer: str, feedback: str, frames: List[str]) -> str:
        content = [{"type": "text", "text": OPTIMIZER_PROMPT.format(
            question=question,
            answer=answer,
            feedback=feedback
        )}]
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })
            
        messages = [{"role": "user", "content": content}]
        improved = self.model(messages).strip()

        return improved

class UrbanTextGrad:
    def __init__(self, qwen_model_name: str = "qwen-vl-max-latest", gemini_model_name: str = "gemini-2.0-flash"):
        self.qwen_model = QwenModel(qwen_model_name)
        self.gemini_model = GeminiModel(gemini_model_name)

        self.reasoner = UrbanReasoner(self.qwen_model)
        self.evaluator = UrbanEvaluator(self.gemini_model)
        self.feedbacker = UrbanFeedbacker(self.qwen_model)
        self.optimizer = UrbanOptimizer(self.qwen_model)
        
    def process(self, question: str, frames: List[str], video_path: str) -> Dict[str, Any]:
        # Initial reasoning
        init_answer = self.reasoner(question, frames)
        init_option = extract_option_letter(init_answer)
        
        # Evaluate current answer
        evaluation = self.evaluator(question, init_answer, frames, video_path)
        eval_option = extract_option_letter(evaluation)
        if init_option == eval_option: # 如果初始答案和评估答案一致，则直接返回
            return {
                "init_answer": init_answer,
                "evaluation": evaluation,
                "feedback": None,
                "final_answer": init_answer,
            }
        
        # Get feedback
        feedback = self.feedbacker(question, init_answer, evaluation)
        
        # Optimize answer
        final_answer = self.optimizer(question, init_answer, feedback, frames)
            
        return {
            "init_answer": init_answer,
            "evaluation": evaluation,
            "feedback": feedback,
            "final_answer": final_answer,
        } 