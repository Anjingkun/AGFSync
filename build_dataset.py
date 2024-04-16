import scorer
import os, json
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from scorer.aesthetic_scorer import MLP, AestheticScorer
from scorer.clip_scorer import CLIPScorer
from scorer.vqa_scorer import VQAScorer
import torch
import numpy as np
from datasets import Dataset
import datasets
import multiprocessing
import time

def score_images(device, scorer_list, args, queue, result_queue):
    # try:
    print("loading model.")
    # device=torch.device(device)
    # model = CLIPModel.from_pretrained(args.clip_path).to(device)
    # model.requires_grad_(False)
    # model.eval()
    # processor = CLIPProcessor.from_pretrained(args.clip_path)
    # aes_model = MLP(768).to(device)
    # aes_model.load_state_dict(torch.load(args.aes_path))
    # aes_model.requires_grad_(False)
    # aes_model.eval()
    # image_processor = CLIPImageProcessor.from_pretrained(args.clip_path)
    with open(args.prompt_path,"r") as file:
        prompts = json.load(file)
    while not queue.empty():
        idx, prompt = queue.get()   
        images = []
        report = {"scorer": {}, "prompt": prompt}
        scores = np.array([0] * args.num_images, dtype=np.float16)
        # for img_id in range(args.num_images):
        #     images.append(Image.open(os.path.join(args.image_dir, f"{idx}_{img_id}.png")))
        for scorer_cls, weight in scorer_list:
            if scorer_cls is AestheticScorer:
                # scorer = scorer_cls(prompt, images, model, image_processor, aes_model)
                prompt = prompts[idx]
                tmp_scores=[]
                for _,score in prompt["images_aes_scores"].items():
                    tmp_scores.append(score)
            elif scorer_cls is CLIPScorer:
                # scorer = scorer_cls(prompt[0:77], images, model, processor)
                prompt = prompts[idx]
                tmp_scores=[]
                for _,score in prompt["images_clip_scores"].items():
                    tmp_scores.append(score)
            elif scorer_cls is VQAScorer:
                # scorer = scorer_cls(prompt, images, idx, args.prompt_path)
                # tmp_scores = scorer.get_score()
                prompt = prompts[idx]
                tmp_scores=[]
                for _,score in prompt["images_vqa_scores"].items():
                    tmp_scores.append(score)
            else:
                raise NotImplementedError("scorer class error.")
            
            tmp_scores = np.array(tmp_scores, dtype=np.float16)
            scores += weight * tmp_scores
        report["win"] = int(np.argmax(scores))
        report["lose"] = int(np.argmax(-scores))
        result_queue.put((idx, report))
        
def main(args):
    # Define Scorer list
    scorer_list = [(CLIPScorer, 0.85), (AestheticScorer, 0.15)]
    # scorer_list = [(VQAScorer, 0.9),(AestheticScorer, 0.01)]
    # scorer_list = [(AestheticScorer, 0.3),(VQAScorer, 0.7)]
    
    # Load prompts
    with open(args.prompt_path, "r") as file:
        prompts = json.load(file)
    
    # Create queues
    gen_queue = multiprocessing.Queue(maxsize=len(prompts))
    score_result_queue = multiprocessing.Queue(maxsize=len(prompts))

    # Populate generation queue
    for prompt in prompts:
        gen_queue.put((prompt["id"], prompt["prompt"]))

    # Start generation processes
    score_processes = []
    for device in args.device.split(","):
        p = multiprocessing.Process(target=score_images, args=(device, scorer_list, args, gen_queue, score_result_queue))
        score_processes.append(p)
        p.start()

    # Wait for scoring processes to finish
    results = {}
    
    progress_bar = tqdm(total=len(prompts))
    while len(results) != len(prompts):
        # print(f"\r {len(results)}/{len(prompts)}")
        # Collect results
        while not score_result_queue.empty():
            id, result = score_result_queue.get()
            results[id] = result
            # Update progress bar
            progress_bar.update(1)
        time.sleep(1)
        
    for p in score_processes:
        p.terminate()
        
    # Gen huggingface dataset
    print(f"dataset size: {len(results)}")
    def gen():
        for key, value in results.items():
            yield {
                "caption": value["prompt"],
                "good_jpg": os.path.join(args.image_dir, f"{key}_{value['win']}.png"),
                "bad_jpg": os.path.join(args.image_dir, f"{key}_{value['lose']}.png"),
            }

    ds = Dataset.from_generator(gen)
    ds = ds.cast_column("good_jpg", datasets.Image())
    ds = ds.cast_column("bad_jpg", datasets.Image())
    ds.save_to_disk(args.output_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_path", type=str, default="models/clip-vit-large-patch14"
    )
    parser.add_argument(
        "--aes_path",
        type=str,
        default="models/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",
    )
    parser.add_argument("--image_dir", type=str, default="dpo_dataset/sdxl_images/")
    parser.add_argument("--output_dir", type=str, default="dpo_dataset/sdxl_clip_aes_dataset_v1/")
    parser.add_argument(
        "--prompt_path", type=str, default="dpo_dataset/prompt/sdxl_prompts_all_scores_results.json"
    )
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--device", type=str, default=",".join(["cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"]*4))
    args = parser.parse_args()

    main(args)
