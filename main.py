from fastapi import FastAPI, Query
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_name, token='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')
tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')

# Check for GPU and move model to GPU if available
device = torch.device("cpu")
model.to(device)


@app.get("/generate")
async def generate(prompt: str = Query('', description="Text prompt for generation")):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# @app.get("/generate")
# async def generate(prompt: str):
#     inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
#     outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}
