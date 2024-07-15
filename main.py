# from fastapi import FastAPI, Request
# from transformers import LlamaForCausalLM, AutoTokenizer
# import torch
#
# # Model and tokenizer loading
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model = LlamaForCausalLM.from_pretrained(model_name, token='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')
# tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')
#
# # Check for GPU and move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# app = FastAPI()
#
#
# @app.get("/generate")
# async def generate(prompt: str):
#     inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
#     outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}
#
#
# if __name__ == '__main__':
#     import uvicorn
#
#     uvicorn.run(app, host='0.0.0.0', port=5000)

"""
server{
       listen 80;
       server_name 92.118.57.15; 
       location / {
           proxy_pass http://127.0.0.1:8081;
       }
}


"""

from fastapi import FastAPI, Request
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

# Model and tokenizer loading
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_name, token='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')
tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')

# Use CPU instead of GPU
device = torch.device("cpu")
model.to(device)

app = FastAPI()


@app.get("/generate")
async def generate(prompt: str):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=5000)
