import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor


print("\n\n******************************\nIMPORTS COMPLETED\n")

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32, # torch.bfloat16 // RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

print("model is loaded!")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
    ]}
]

print("wrote the images and message!")

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

print("running model prediction!")

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))


print("done!")


'''
******************************
IMPORTS COMPLETED

The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████| 5/5 [00:06<00:00,  1.35s/it]
model is loaded!
wrote the images and message!
running model prediction!
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>If I had to write a haiku for this one, it would be: <|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here is a haiku for the image:

Rabbit in a coat
Walking down a dirt path
Springtime delight<|eot_id|>
done!
'''