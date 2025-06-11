from typing import List, Dict, Any
from prompts import (
    REASONER_PROMPT, 
    EVALUATOR_PROMPT, 
    FEEDBACKER_PROMPT, 
    OPTIMIZER_PROMPT,
    OPTIMIZER_SYSTEM_PROMPT
)
from src.clients.qwen_client import qwen_client

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
    def __init__(self, model: QwenModel):
        self.model = model
        
    def __call__(self, question: str, answer: str, frames: List[str]) -> str:
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
            answer=answer,
            feedback=feedback
        )}]
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })
            
        messages = [{"role": "user", "content": content}]
        improved = self.model(messages, system_prompt=OPTIMIZER_SYSTEM_PROMPT)
        # Extract the improved answer from between the tags
        try:
            improved = improved.split("<IMPROVED_VARIABLE>")[1].split("</IMPROVED_VARIABLE>")[0].strip()
        except:
            improved = improved.strip()
        return improved

class UrbanTextGrad:
    def __init__(self, model_name: str = "qwen-vl-max-latest"):
        self.model = QwenModel(model_name)
        self.reasoner = UrbanReasoner(self.model)
        self.evaluator = UrbanEvaluator(self.model)
        self.feedbacker = UrbanFeedbacker(self.model)
        self.optimizer = UrbanOptimizer(self.model)
        
    def process(self, question: str, frames: List[str]) -> Dict[str, Any]:
        # Initial reasoning
        init_answer = self.reasoner(question, frames)
        
        # Evaluate current answer
        evaluation = self.evaluator(question, init_answer, frames)
        
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