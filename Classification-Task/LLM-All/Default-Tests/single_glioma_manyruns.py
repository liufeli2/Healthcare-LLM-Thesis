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

def synthesize_response(response):
    parts = response.split("<|end_header_id|>")
    cleaned_response = parts[-1].replace("\n", " || ").replace("*", "").strip()
    match = re.search(r"(Choice:\s*\d+\s+Reasoning:.*?)(?=<|$)", cleaned_response, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()  # Capture the full match
        choice_match = re.search(r"Choice:\s*(\d+)", extracted_text)
        if choice_match:
            choice = choice_match.group(1)
            return extracted_text, int(choice)
    return "ERROR: Unknown >>"+cleaned_response, 3


def tally_count(vector):
    return [(vector==num).sum().item() for num in [0, 1, 2, 3]]


def extract_llm_predictions(num, number_runs, image_path):
    print(f"\nScan #: {num}/{class_map[num]}")
    for i in range(1):
        with open(f'scan_results_{num}.txt', 'w') as file:
            file.write(f"Scan #: {num}/{class_map[num]}\n")
            guesses = []
            for slice_i in range(number_runs):
                text, response = predict_one_picture(image_path)
                file.write(f"Slice #{slice_i} => {text}\n")
                print(f"Slice #{slice_i} => {text}")
                guesses.append(response)
            file.write(f"Guesses: {guesses}\n")
            print(f"Guesses: {guesses}")
            tally = tally_count(np.array(guesses))
            file.write(f"Tally: {tally}\n")
            print(f"Tally: {tally}")
            file.write(f"True Label: {class_map[num]} || Prediction Label: {class_map[np.argmax(tally[0:2])]}\n\n")
            print(f"True Label: {class_map[num]} || Prediction Label: {class_map[np.argmax(tally[0:2])]}\n")


################################################################################# 
# MAIN
################################################################################# 

class_map = {"LGG":0, "HGG":1, "NG":2, "Unknown":3, 0:"LGG", 1:"HGG", 2:"NG", 3:"Unknown"}
model, processor, messages = setup_glioma_predictor()

image_path_hgg = "/scratch/k/khalvati/liufeli2/LLMs/data/images/highgrade_test.png"
image_path_lgg = "/scratch/k/khalvati/liufeli2/LLMs/data/images/lowgrade_test.png"

try:
    extract_llm_predictions(num=0, number_runs=95, image_path=image_path_lgg)
    extract_llm_predictions(num=1, number_runs=95, image_path=image_path_hgg)

except KeyboardInterrupt:
    print("Process interrupted. Clearing memory...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Memory cleared successfully.")


print("\nscript done!\n******************************")
################################################################################# 
# RESULTS
################################################################################# 