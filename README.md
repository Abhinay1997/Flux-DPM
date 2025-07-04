```python
import torch
from flux_pipeline_dpm import FluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler



pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16)
euler_scheduler = pipe.scheduler
dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained('Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers', subfolder='scheduler')

pipe.to("cuda")


## baseline
pipe.scheduler = euler_scheduler
out = pipe(
    prompt="An oil painting of a light house in a stormy sea, ships visible in the distance, lightning in the sky. 18th century style",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=28,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

## DPM soheduler
pipe.scheduler = dpm_scheduler
out_dpm = pipe(
    prompt="An oil painting of a light house in a stormy sea, ships visible in the distance, lightning in the sky. 18th century style",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=18,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
```

```python
import torch
from flux_kontext_pipeline_dpm import FluxKontextPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
euler_scheduler = pipe.scheduler
dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained('Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers', subfolder='scheduler')

pipe.to("cuda")

input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
## Baseline
pipe.scheduler = euler_scheduler
image = pipe(
  image=input_image,
  prompt="Add a hat to the cat",
  guidance_scale=2.5,
  num_inference_steps=28,
).images[0]

## DPM
pipe.scheduler = dpm_scheduler
image_dpm = pipe(
  image=input_image,
  prompt="Add a hat to the cat",
  guidance_scale=2.5,
  num_inference_steps=20,
).images[0]
```

```python
import torch 
from flux_kontext_inpaint_pipeline_dpm import FluxKontextInpaintPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image

prompt = "Change the yellow dinosaur to green one"
pipe = FluxKontextInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to('cuda')

img_url = "https://github.com/ZenAI-Vietnam/Flux-Kontext-pipelines/blob/main/assets/dinosaur_input.jpeg?raw=true"
mask_url = "https://github.com/ZenAI-Vietnam/Flux-Kontext-pipelines/blob/main/assets/dinosaur_mask.png?raw=true"

source = load_image(img_url)
mask = load_image(mask_url)

euler_scheduler = pipe.scheduler
dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained('Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers', subfolder='scheduler')

## baseline
pipe.scheduler = euler_scheduler
image = pipe(
    prompt=prompt, 
    image=source,
    mask_image=mask,
    strength=1.0,
    num_inference_steps=28,
).images[0]

## dpm
pipe.scheduler = dpm_scheduler
image_dpm = pipe(
    prompt=prompt, 
    image=source,
    mask_image=mask,
    strength=1.0,
    num_inference_steps=20,
).images[0]
```
