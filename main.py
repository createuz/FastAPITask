from fastapi import FastAPI, Request
import uvicorn
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

# Загрузка модели и токенизатора
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_name, token='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')
tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')

app = FastAPI()


@app.get('/generate')
async def generate(prompt: str):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
