################################################################################# 
# IMPORTS
################################################################################# 

import warnings
warnings.filterwarnings("ignore", message=".*torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta.*")

import os
import re
from PIL import Image
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import numpy as np

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.set_per_process_memory_fraction(0.8, 0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

print("******************************\nIMPORTS COMPLETED\n")
################################################################################# 
# FUNCTIONS
################################################################################# 

def unload_data_from_npz(split_data_path, split_segs_path):
    loaded_data = np.load(split_data_path, allow_pickle=True)
    loaded_segs = np.load(split_segs_path, allow_pickle=True)
    voxels = loaded_data['voxels']
    labels = loaded_data['labels']
    segs = loaded_segs['segs']
    tvoxels = loaded_data['tvoxels']
    tlabels = loaded_data['tlabels']
    tsegs = loaded_segs['tsegs']
    num_folds = (len(loaded_data.files)-4)//2
    train_inds = [loaded_data[f'train_inds_{i}'] for i in range(num_folds)]
    val_inds = [loaded_data[f'val_inds_{i}'] for i in range(num_folds)]
    print(f"TRAIN/VAL DATA: {voxels.shape} {segs.shape} {labels.shape}")
    print(f"TEST DATA:      {tvoxels.shape} {tsegs.shape} {tlabels.shape}")
    for i in range(len(train_inds)):
        print(f"fold #{i}: train/val [{len(train_inds[i])}/{len(val_inds[i])}]")
    return voxels, segs, labels, tvoxels, tsegs, tlabels, num_folds, train_inds, val_inds


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


def predict_one_picture(np_array):
    # open the given image
    image = save_and_open_temp_image(np_array, "temp_image.png")
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
    # help with memory
    del inputs, image, output
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
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


def extract_llm_predictions(voxels, segs, labels, starting_ind=0, type_slice="axial", flair_only=False):

    # change which layout of slices we're working with!
    if type_slice == "axial": print("axial!"); voxels = voxels
    elif type_slice == "coronal": print("coronal!"); voxels = np.transpose(voxels, (0, 2, 1, 3))
    elif type_slice == "sagittal": print("sagittal!"); voxels = np.transpose(voxels, (0, 3, 1, 2))
    else: print("no mode found!"); return

    total_scans = len(voxels)
    for i, (scan, label, seg) in enumerate(zip(voxels, labels, segs)):
        print(f"\nScan #: {i}/{total_scans}")

        # get the scan slices
        start_slice, end_slice = seg
        print(f"Indices: {start_slice}, {end_slice}")

        if i < starting_ind:
            print("Continuing because DONE")
            continue

        if flair_only and i % 4 != 0: 
            print("Continuing because NOT FLAIR")
            continue

        with open(f'scan_results_{i}.txt', 'w') as file:
            file.write(f"Scan #: {i}/{total_scans}\n")
            guesses = []
            for slice_i, axial_slice in enumerate(scan):
                # only look at slices in the range
                if slice_i < start_slice: continue
                if slice_i > end_slice: break
                # if in the range, then we calculate
                text, response = predict_one_picture(axial_slice)
                file.write(f"Slice #{slice_i} => {text}\n")
                print(f"Slice #{slice_i} => {text}")
                guesses.append(response)
            file.write(f"Guesses: {guesses}\n")
            print(f"Guesses: {guesses}")
            tally = tally_count(np.array(guesses))
            file.write(f"Tally: {tally}\n")
            print(f"Tally: {tally}")
            file.write(f"True Label: {class_map[label]} || Prediction Label: {class_map[np.argmax(tally[0:2])]}\n\n")
            print(f"True Label: {class_map[label]} || Prediction Label: {class_map[np.argmax(tally[0:2])]}\n")


def save_and_open_temp_image(np_array, temp_path):
    # used to be just using the np array
    # image = Image.fromarray(np_array)
    # now saves as a image and then reuploads as a photo
    np_array = np.uint8(255 * (np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array)))
    image = Image.fromarray(np_array)
    image.save(temp_path)
    image_opened = Image.open(temp_path)
    try:
        os.remove(temp_path)
    except Exception as e:
        print(f"Warning: Could not delete temporary file {temp_path}. Error: {e}")
    return image_opened


################################################################################# 
# MAIN
################################################################################# 

model, processor, messages = setup_glioma_predictor()

# reload from the files of split data
split_data_path = "/scratch/k/khalvati/liufeli2/LLMs/data/SPLIT_DATA_FULL.npz"
split_segs_path = "/scratch/k/khalvati/liufeli2/LLMs/data/SPLIT_SEGS_CORONAL_FULL.npz"
voxels, segs, labels, tvoxels, tsegs, tlabels, num_folds, train_inds, val_inds = unload_data_from_npz(split_data_path, split_segs_path)
class_map = {"LGG":0, "HGG":1, "NG":2, "Unknown":3, 0:"LGG", 1:"HGG", 2:"NG", 3:"Unknown"}

# define params
type_slice = "sagittal" # choose axial, coronal sagittal
flair_only = True       # boolean indicates if 
starting_ind = 104      # for the timeout, to restart where we left off

try:
    extract_llm_predictions(tvoxels, tsegs, tlabels, starting_ind=starting_ind,
                            type_slice=type_slice, flair_only=flair_only)


except KeyboardInterrupt:
    print("Process interrupted. Clearing memory...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Memory cleared successfully.")




print("\nscript done!\n******************************")
################################################################################# 
# RESULTS
################################################################################# 