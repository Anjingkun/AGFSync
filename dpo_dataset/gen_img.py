from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, LCMScheduler
import torch
import json
from multiprocessing import Process, Queue, Manager, set_start_method
import argparse
import os
from compel import Compel, ReturnedEmbeddingsType
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def dummy(images, **kwargs): 
	return images, [False] * len(images)

def gen(device, args, queue):
    # 加载Stable Diffusion
    pipeline_cls = StableDiffusionPipeline if args.sd_version != "xl" else StableDiffusionXLPipeline
    if args.sd_version == "1.5":
        pipe = pipeline_cls.from_pretrained(args.model_path,
                                        variant="fp16",
                                        torch_dtype=torch.float16,
                                        safety_checker=dummy,
                                        cache_dir = "../models/sd15"
                                    )
    elif args.sd_version == "xl":
        pipe = pipeline_cls.from_pretrained(args.model_path,
                                        variant="fp16",
                                        torch_dtype=torch.float16,
                                        safety_checker=dummy,
                                        cache_dir="../models/sdxl"
                                    )
    pipe.to(device)
    # 开启xformers加速
    pipe.enable_xformers_memory_efficient_attention()
    
    # 加载LCM LoRA，提高数据集生成速度
    if args.lcm_path is not None:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(args.lcm_path)
        pipe.fuse_lora()
        
    while not queue.empty():
        idx, prompt = queue.get()
        if all([os.path.exists(os.path.join(args.output_dir, f"{idx}_{count}.png")) for count in range(args.num_images)]):
            print(f"skip {idx}")
            continue
        
        prompt_kwargs = {}
        if args.sd_version == "xl":
            compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
            conditioning, pooled = compel(prompt)
            pooled = pooled.repeat((args.num_images,1))
            pooled[1:] = pooled[1:] + args.noise_ratio * torch.randn_like(pooled[1:])
            prompt_kwargs["pooled_prompt_embeds"] = pooled + torch.randn_like(pooled)
        else:
            compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            conditioning = compel(prompt)
            
        conditioning = conditioning.repeat((args.num_images,1,1))
        conditioning[1:] = conditioning[1:] + args.noise_ratio * torch.randn_like(conditioning[1:])
        prompt_kwargs["prompt_embeds"] = conditioning
        
        num_inference_steps = 5 if args.lcm_path is not None else 15
        guidance_scale= 1.0 if args.lcm_path is not None else 4.5
        with torch.no_grad():
            images = pipe(
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=[torch.Generator()]*args.num_images,
                **prompt_kwargs
            ).images
        
        for img_idx, image in enumerate(images):
            image.save(os.path.join(args.output_dir, f"{idx}_{img_idx}.png"))

if __name__ == "__main__":
    set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--device", type=str, default=",".join(["cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"]*1))
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt_path", type=str, default="prompt/prompts_qas.json")
    parser.add_argument("--lcm_path", type=str, default="latent-consistency/lcm-lora-sdxl")
    parser.add_argument("--output_dir", type=str, default="../dpo_dataset/sdxl_images_test")
    parser.add_argument("--noise_ratio", type=float, default=0.1)
    parser.add_argument("--sd_version", type=str, default="xl")
    
    args = parser.parse_args()
    
    with open(args.prompt_path, "r") as file:
        prompts = json.load(file)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    manager = Manager()
    queue = manager.Queue()
    for prompt in prompts:
        queue.put((prompt["id"], prompt["prompt"]))

    processes = []
    for device in args.device.split(","):
        p = Process(target=gen, args=(device, args, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
