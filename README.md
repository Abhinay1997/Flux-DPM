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
```python
import torch
from pipeline_qwenimage_dpm import QwenImagePipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
pipe.to('cuda')

euler_scheduler = pipe.scheduler
dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained('Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers', subfolder='scheduler')
prompt = """A cozy bookstore window on a charming street, framed with lush greenery—vibrant ivy cascading around the edges and potted ferns spilling onto the sidewalk. The window’s glass catches the warm, golden glow of late afternoon sunlight, casting soft reflections of the surrounding street, including faint outlines of passing clouds and nearby trees. Inside, two bestselling books are prominently displayed on a velvet-lined shelf: “The Night Circus” by Erin Morgenstern, its cover featuring a striking black-and-white circus tent under a starry sky with a bold red scarf swirling across it, and “Where the Crawdads Sing” by Delia Owens, showcasing a serene marsh landscape with a lone canoe and a vibrant sunset in hues of orange and pink. The books are angled to catch the light, their glossy covers glinting subtly. Outside, a fluffy tabby cat with amber eyes sits on the cobblestone street, delicately licking its paw, while a monarch butterfly with vivid orange and black wings flutters past, catching the light. A man in a tweed jacket stands in front of the window, gazing intently at the books, his face etched with deep thought, his reflection faintly visible in the glass. The scene is bathed in a mix of warm sunlight and cool shadows, with the bookstore’s interior glowing softly from within, creating a layered interplay of light and reflection on the window."""

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt,
    "zh": "超清，4K，电影级构图" # for chinese prompt,
}

# Generate image
negative_prompt = " "


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}

width, height = aspect_ratios["4:3"]

pipe.scheduler = euler_scheduler
image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
image.save("example.jpg")

pipe.scheduler = dpm_scheduler
image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=28,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]
image.save("example_dpm.png")
```

if you find this useful, consider citing as:
```
@misc{flux-dpm,
    author={Naga Sai Abhinay Devarinti},
    title={Flux-DPM},
    year={2025},
    howpublished={\url{https://github.com/Abhinay1997/Flux-DPM}},
}
```
Citations:
```
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}

@misc{flux2024,
    author={Black Forest Labs},
    title={FLUX},
    year={2024},
    howpublished={\url{https://github.com/black-forest-labs/flux}},
}

@misc{xie2025sana,
      title={SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer},
      author={Xie, Enze and Chen, Junsong and Zhao, Yuyang and Yu, Jincheng and Zhu, Ligeng and Lin, Yujun and Zhang, Zhekai and Li, Muyang and Chen, Junyu and Cai, Han and others},
      year={2025},
      eprint={2501.18427},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.18427},
    }
```
