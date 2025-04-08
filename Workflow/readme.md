# Llama-3.2-11B-Vision-Instruct

Model of Interest: [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)


Llama 3.2-Vision is a collection of multimodal large language models (11B and 90B) optimized for visual recognition, image reasoning, captioning, and answering image-related questions. Built on the Llama 3.1 architecture, it integrates a vision adapter with cross-attention layers to process image data, fine-tuned using supervised learning and reinforcement learning with human feedback.

> **Benefits:** Image identification is already worked into the model and we are able to prompt the model to answers. Easier for my initial run to work with this.

> **Concerns:** Unlike Clinical Camel, this model has not been trained in any capacity on any form a specific medical data, hence we have no idea what the baseline performance of this model could be.


## SciNet - Setup Instructions
> This part was very painful to setup... do it carefully!


**Get Permissions and Login to SciNet**
- SciNet Account: https://www.scinethpc.ca/getting-a-scinet-account/
- Niagara (CPU Cluster) Documentation: https://docs.scinet.utoronto.ca/index.php/Niagara_Quickstart
- MIST Documentation: https://docs.scinet.utoronto.ca/index.php/Mist#PyTorch
```
ssh -i /path/to/ssh_private_key -Y MYCCUSERNAME@mist.scinet.utoronto.ca

ssh -i /path/to/ssh_private_key -Y MYCCUSERNAME@niagara.scinet.utoronto.ca
ssh -Y mist-login01
```


**Move to MIST and SCRATCH Directory**
```
ssh -Y mist-login01
cd $SCRATCH
pwd 

// check you are in your scratch directory on mist
```

**Load Module Spider Units**
``` 
module load anaconda3/2021.05 
module load cuda/11.0.3 
module load gcc/8.5.0
module list

// make sure to have: 1) MistEnv/2021a (S)   2) anaconda3/2021.05   3) cuda/11.0.3   4) gcc/8.5.0
```

**Everything I Ran to Setup Conda Environment**
``` 
conda create -n llms python=3.9
source activate llms

conda config --prepend channels /scinet/mist/ibm/open-ce-1.9.1
conda config --set channel_priority strict

conda install -c /scinet/mist/ibm/open-ce-1.9.1 pytorch=2.0.1 cudatoolkit=11.8

conda install transformers=4.32.1 -y
conda remove transformers --force-remove -y
pip install transformers
python -m pip install "accelerate>=0.26.0"

// this should match req5.txt environment
```


## Setup Video and Links

- Video: https://www.youtube.com/watch?v=zGqQGtmXFQ8
- Uses CUDA 12.4 (11.7 CUDA on my SciNet system)
- Uses HugginFace (Which I previously already signed up for! I did need to request access for the model though Llama-3.2-11B-Vision-Instruct)


## 3.2 Vision or 3.2 Vision Instruct

From what I can tell, they have virtually the same description and usability but I assume the Instruct Model generates responses more with the direct prompts, hence, we will start with Llama-3.2-11B-Vision-Instruct. It also seems to be used in more tutorials, and currently has more downloads. Getting permissions for 1 gives you permissions for both so this can be changed easily in the future. There's also a 90B parameter but I think I will start with the smaller one!

**Starting Code Provided**
``` 
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
```

## Really Useful Currently

**Using MIST Conda Environment Everytime After**
```
// copy this into terminal line by line
ssh -Y mist-login01
cd $SCRATCH
pwd 
module load anaconda3/2021.05 cuda/11.0.3 gcc/8.5.0
module list
conda activate llms

// sometimes, activate just doesn't work and pytorch can't be imported
conda deactivate
conda deactivate
conda activate llms
// fixes it for reasons I don't understand (resets env?)
```

**Memory Issues and How to Resolve Them**
```
// cmd+C and cmd+X interupt the process and backlog unfinished processes
// check what's still consuming space and 'kill' them directly (-9, -15)
nvidia-smi
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   3755773      C   python                          17303MiB |
|    1   N/A  N/A   3755773      C   python                          12007MiB |
|    2   N/A  N/A   3755773      C   python                          12209MiB |
|    3   N/A  N/A   3755773      C   python                          11645MiB |
+-----------------------------------------------------------------------------+
kill -15 3755773

// clear things manually when interupted to free memory
try:
    while True:  
        # code for long-running process here!
except KeyboardInterrupt:
    print("Process interrupted. Clearing memory...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Memory cleared successfully.")

// other memory helpers
torch_dtype=torch.float32 -> torch.float16
del inputs, image, output -> stuff you don't need after use
```

**SciKit-Learn and Transformers Conflict**
```
// Successfully uninstalled huggingface-hub-0.17.3
// Successfully installed huggingface-hub-0.26.2

// scikit-learn and transformers might have different huggingface requirements
python -m pip install transformers
conda install scikit-learn

// just change the download to make it work!
```


## Random Info

```
final                    /home/k/khalvati/liufeli2/.conda/envs/final
llm117                   /home/k/khalvati/liufeli2/.conda/envs/llm117
llm32                    /home/k/khalvati/liufeli2/.conda/envs/llm32
llm38                    /home/k/khalvati/liufeli2/.conda/envs/llm38
llms                     /home/k/khalvati/liufeli2/.conda/envs/llms
myenv                    /home/k/khalvati/liufeli2/.conda/envs/myenv
new                      /home/k/khalvati/liufeli2/.conda/envs/new
pytorch                  /home/k/khalvati/liufeli2/.conda/envs/pytorch
tf_env                   /home/k/khalvati/liufeli2/.conda/envs/tf_env
base                  *  /scinet/mist/rhel8/software/2021a/opt/base/anaconda3/2022.05
```

```
python              3.11        3.9         3.10          
torch               2.4.0       1.12.1      1.13.1
tensorboard                     2.8.0
pillow                          8.4.0 
torchvision                     0.13.0
transformers        4.45.1      4.19.0
datasets            3.0.1       2.4.0
accelerate          0.34.2      0.6.0
evaluate            0.4.3       0.2.2
bitsandbytes        0.44.0      0.39.0
trl                 0.11.1      0.4.0
peft                0.13.0      0.1.0
```