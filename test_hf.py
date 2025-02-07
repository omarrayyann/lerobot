from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

model_id = "lerobot/pi0"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

model = PI0Policy.from_pretrained(model_id).model.paligemma_with_expert.paligemma.eval()
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

# Instruct the model to create a caption in Spanish
prompt = "whats the color car?"
model_inputs = processor(text=prompt, images=image, return_tensors="pt")
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    print(generation)
    generation = generation[0][input_len:]
    print(generation)
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)