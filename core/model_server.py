import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentienceModelServer")

app = FastAPI(title="Sentience Core Inference Server")

# Global variables for model and tokenizer
model = None
tokenizer = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 500
    stream: bool = False

@app.on_event("startup")
def load_model():
    global model, tokenizer
    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "ml/models/sentience_core_lora"
    
    logger.info(f"Loading base model: {base_model_id}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    try:
        # Load base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load adapter if it exists
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            logger.info(f"Applying LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            logger.warning("No adapter found. Using base model only.")
            model = base_model
            
        logger.info("Sentience Core Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model, tokenizer
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Format the prompt using ChatML-like structure for Phi-3
        prompt = ""
        for msg in request.messages:
            role = "user" if msg.role == "user" else "assistant"
            prompt += f"<|{role}|>\n{msg.content}<|end|>\n"
        prompt += "<|assistant|>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True if request.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {
            "id": "sentience-123",
            "object": "chat.completion",
            "created": 123456789,
            "model": "sentience-core-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/model/reload")
async def reload_adapter():
    global model
    adapter_path = "ml/models/sentience_core_lora"
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            logger.info(f"Hot-reloading LoRA adapter from: {adapter_path}")
            # Loading new adapter weights into the existing model
            if isinstance(model, PeftModel):
                model.unload() # Clear old adapter
            model = PeftModel.from_pretrained(model.get_base_model(), adapter_path)
            return {"status": "success", "message": "Adapter reloaded"}
        else:
            return {"status": "error", "message": "No adapter found to reload"}
    except Exception as e:
        logger.error(f"Reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
