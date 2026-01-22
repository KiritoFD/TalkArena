import re
import gc
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from .structs import EmotionMetadata, AggressionLevel


class LLMEngine:
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.DEFAULT_MODEL
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self._loaded = True
    
    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate(
        self,
        user_input: str,
        system_prompt: str,
        history: Optional[list] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        if not self._loaded:
            self.load()
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            for turn in history[-10:]:
                role = "assistant" if turn.get("is_ai") else "user"
                messages.append({"role": role, "content": turn["text"]})
        
        messages.append({"role": "user", "content": user_input})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    @staticmethod
    def parse_emotion(text: str) -> Tuple[str, EmotionMetadata]:
        """Parse emotion metadata from LLM output and return clean text."""
        emotion = EmotionMetadata()
        clean_text = text
        
        aggression_match = re.search(r'\[Aggression:\s*(Low|Medium|High)\]', text, re.IGNORECASE)
        if aggression_match:
            level = aggression_match.group(1).capitalize()
            emotion.aggression_level = AggressionLevel(level)
            clean_text = clean_text.replace(aggression_match.group(0), "")
        
        confidence_match = re.search(r'\[Confidence:\s*(\d+)\]', text, re.IGNORECASE)
        if confidence_match:
            emotion.confidence_level = min(100, max(0, int(confidence_match.group(1))))
            clean_text = clean_text.replace(confidence_match.group(0), "")
        
        stress_match = re.search(r'\[Stress:\s*(\d+)\]', text, re.IGNORECASE)
        if stress_match:
            emotion.stress_level = min(100, max(0, int(stress_match.group(1))))
            clean_text = clean_text.replace(stress_match.group(0), "")
        
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text, emotion


EMOTION_INSTRUCTION = """
在每次回复的末尾，添加以下情感标记（必须添加，用于系统分析）：
[Aggression: Low/Medium/High] [Confidence: 0-100] [Stress: 0-100]

示例：
"我觉得你说得有道理，但是这个方案还需要再考虑。[Aggression: Low] [Confidence: 60] [Stress: 30]"
"你这样说是什么意思？你觉得我做不到吗？[Aggression: High] [Confidence: 70] [Stress: 50]"
"""
