import sys 
sys.path.append("")
import os, json
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from scorer.clip_scorer import CLIPScorer
import torch
import numpy as np
from datasets import Dataset
import datasets
import multiprocessing
import time

def score_images(device, scorer_list, args, queue, result_queue):
    # try:
    print("loading model.")
    device=torch.device(device)
    model = CLIPModel.from_pretrained(args.clip_path).to(device)
    model.requires_grad_(False)
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.clip_path)

    while not queue.empty():
        data = queue.get()
        idx = data["id"]
        prompt = data["prompt"]
        images = []
        data["images_clip_scores"]={}

        for img_id in range(args.num_images):
            images.append(Image.open(os.path.join(args.image_dir, f"{idx}_{img_id}.png")))
        for scorer_cls in scorer_list:
            if scorer_cls is CLIPScorer:
                scorer = scorer_cls(prompt[0:77], images, model, processor)
                for image_index, score in enumerate(scorer.get_score()):
                    data["images_clip_scores"][f"{idx}_{image_index}.png"] = score
            else:
                raise NotImplementedError("scorer class error.")
        result_queue.put(data)
        
def main(args):
    # Define Scorer list
    scorer_list = [CLIPScorer]
    
    # Load prompts
    with open(args.prompt_path, "r") as file:
        prompts = json.load(file)
    
    # Create queues
    gen_queue = multiprocessing.Queue(maxsize=len(prompts))
    score_result_queue = multiprocessing.Queue(maxsize=len(prompts))

    # Populate generation queue
    for prompt in prompts:
        gen_queue.put(prompt)

    # Start generation processes
    score_processes = []
    for device in args.device.split(","):
        p = multiprocessing.Process(target=score_images, args=(device, scorer_list, args, gen_queue, score_result_queue))
        score_processes.append(p)
        p.start()

    # Wait for scoring processes to finish
    results = []
    
    progress_bar = tqdm(total=len(prompts))
    while len(results) != len(prompts):
        # print(f"\r {len(results)}/{len(prompts)}")
        # Collect results
        while not score_result_queue.empty():
            result = score_result_queue.get()
            results.append(result)
            # Update progress bar
            progress_bar.update(1)
        time.sleep(1)
        if len(results)%10==0:
            with open(args.out_file, 'w') as file:
                json.dump(results, file, indent=4)
        
    for p in score_processes:
        p.terminate()
    results = sorted(results, key=lambda d: d['id'])
    with open(args.out_file, 'w') as file:
        json.dump(results, file, indent=4)
        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_path", type=str, default="../models/clip-vit-large-patch14"
    )
    parser.add_argument("--image_dir", type=str, default="sdxl_images/")
    parser.add_argument(
        "--prompt_path", type=str, default="prompt/sdxl_prompts_vqa_score.json"
    )
    parser.add_argument(
        "--out_file", type=str, default='prompt/sdxl_prompts_vqa_clip_scores_results.json'
    )
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--device", type=str, default=",".join(["cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:6,cuda:7"]*4))
    args = parser.parse_args()

    main(args)
