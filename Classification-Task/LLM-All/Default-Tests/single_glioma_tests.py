################################################################################# 
# IMPORTS
################################################################################# 

import warnings
warnings.filterwarnings("ignore", message=".*torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta.*")

import re
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

import numpy as np

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.set_per_process_memory_fraction(0.8, 0)

print("******************************\nIMPORTS COMPLETED\n")
################################################################################# 
# FUNCTIONS
################################################################################# 

def setup_glioma_predictor():
    # get the model downloaded from shards
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    print("model is loaded!\n")

    # write the general message prompt
    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Classify the brain scan as Low Grade Glioma (0), High Grade Glioma (1), or No Glioma (2). Respond only in the following format: Choice: <0, 1, or 2> Reasoning: <Provide concise reasoning using 10 keywords based on the scan's visual features>."}
    ]}]
    return model, processor, messages

def predict_one_picture(image_path):
    # open the given image
    image = Image.open(image_path)
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    # have the model generate a response
    output = model.generate(**inputs, max_new_tokens=20)
    response = processor.decode(output[0])
    return synthesize_response(response)

def extract_cleaned_content(response):
    parts = response.split("<|end_header_id|>", 1)
    if len(parts) > 1:
        cleaned_content = parts[1].replace("\n", "").strip()
        return cleaned_content
    return "" 

def synthesize_response(response):
    # Regular expression to match "Choice: <number> Reasoning: <text>"
    match = re.search(r"(Choice:\s*\d+\s+Reasoning:.*?)(?=<|$)", response, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()  # Capture the full match
        # print("Extracted Section:", extracted_text)  # Print the extracted section
        choice_match = re.search(r"Choice:\s*(\d+)", extracted_text)
        if choice_match:
            choice = choice_match.group(1)
            # print(f"Extracted Choice: {choice}")
            return extracted_text, int(choice)
    return "ERROR: Unknown >>"+extract_cleaned_content(response), 3


################################################################################# 
# MAIN
################################################################################# 

model, processor, messages = setup_glioma_predictor()

image_path = "/scratch/k/khalvati/liufeli2/LLMs/data/images/highgrade_test.png"
reponse = predict_one_picture(image_path)
print(reponse)

image_path = "/scratch/k/khalvati/liufeli2/LLMs/data/images/lowgrade_test.png"
reponse = predict_one_picture(image_path)
print(reponse)


print("\nscript done!\n******************************")
################################################################################# 
# RESULTS
################################################################################# 

'''

######################
Analyze the brain scan and classify the presence of a glioma based on visible 
features in the image. Determine if it is a low-grade glioma (LLG), high-grade 
glioma (HGG), or no glioma present (NG).

**Step 1: Identify the visible features in the image.**
The image shows a brain scan with a mass lesion in the left frontal lobe


######################
Classify brain scan and if a low-grade glioma (LLG), high-grade glioma (HGG), or 
no glioma (NG), is present.

To classify the brain scan, we need to analyze the image and identify any abnormalities 
that could indicate the presence of a glioma. 


######################
Classify brain scan as 'LGG' (low-grade glioma), 'HGG' (high-grade glioma), or 'NG' (no 
glioma present).

To classify the brain scan, we need to analyze the characteristics of the tumor and its appearance on the scan.
**Relevant Data:**


######################
Respond only by selecting 1, 2, or 3. Classify brain scan as 'LGG' (1 - low-grade glioma), 
'HGG' (2 - high-grade glioma), or 'NG' (3 - no glioma present).

To classify the brain scan, we need to analyze the image and identify any characteristic 
features of low-grade gliomas (LGG), high-grade gliomas


######################
Classify the brain scan as 'LGG' (1 - low-grade glioma), 'HGG' (2 - high-grade glioma), 
or 'NG' (3 - no glioma present). Respond only with 1, 2, or 3.

2.


######################
Classify the brain scan as 'LGG' (0 - low-grade glioma), 'HGG' (1 - high-grade glioma), or 
'NG' (2 - no glioma present) based on visual features present. Respond only with 0, 1, or 2, 
no need to provide explanation.

1. 

^^ but noticed this one always only responded with 1... or 2 sometimes



Classify the brain scan as Low Grade Glioma (0), High Grade Glioma (1), or No Glioma (2). 
- Low Grade Gliomas typically appear well defined with minimal disruption to surrounding tissue.
- High Grade Gliomas often show irregular enhancement, necrosis, and significant edema.
Examine features carefully and respond with 0, 1, or 2.

Classify the brain scan as Low Grade Glioma (0), High Grade Glioma (1), or No Glioma (2). Respond only in the following format: Choice: <0, 1, or 2> Reasoning: <Provide reasoning in one concise sentence based on the scan's visual features>.


High Grade Gliomas show irregular enhancement, necrosis, and significant edema.
Low Grade Gliomas typically appear well defined with minimal disruption to surrounding tissue.

******************************
'''