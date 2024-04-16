import json
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import multiprocessing
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import time, tqdm


def process_prompt(task_queue, result_queue, device, clip_path):
    # 初始化模型和处理器
    processor = Blip2Processor.from_pretrained(clip_path)
    model = Blip2ForConditionalGeneration.from_pretrained(clip_path, torch_dtype=torch.float16,
                                                          device_map=int(device.split(':')[1]),
                                                          cache_dir="../models/blip2")

    while not task_queue.empty():

        task, img_folder_path = task_queue.get()

        prompt_id = task["id"]
        for qa_pair in task["qa_pairs"]:
            question = qa_pair['question']
            prompt = f"Question: {question} Choices:['yes','no'] Answer:"
            qa_pair["images_answers"] = {}
            for i in range(8):
                img_path = os.path.join(img_folder_path, f'{prompt_id}_{i}.png')
                raw_image = Image.open(img_path).convert('RGB')
                inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device, torch.float16)
                outputs = model.generate(**inputs, max_new_tokens=512)
                answer = processor.decode(outputs[0], skip_special_tokens=True)
                qa_pair["images_answers"][f"{prompt_id}_{i}.png"] = answer
        result_queue.put(task)


def main():
    clip_path = "Salesforce/blip2-flan-t5-xxl"
    img_folder_path = 'sdxl_images'
    devices = ",".join(["cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"] * 2)
    prompt_path = 'prompt/prompts_qas.json'

    with open(prompt_path, 'r') as file:
        data = json.load(file)

    task_queue = multiprocessing.Queue(maxsize=len(data))
    result_queue = multiprocessing.Queue(maxsize=len(data))

    for item in data:
        task_queue.put((item, img_folder_path))

    processes = []

    for device in devices.split(","):
        p = multiprocessing.Process(target=process_prompt, args=(task_queue, result_queue, device, clip_path))
        processes.append(p)
        p.start()

    # Initialize tqdm progress bar
    progress_bar = tqdm.tqdm(total=len(data))
    results = []
    while len(results) != len(data):
        # print(f"\r {len(results)}/{len(data)}")
        while not result_queue.empty():
            item = result_queue.get()
            results.append(item)
            # Update progress bar
            progress_bar.update(1)
        time.sleep(1)
        if len(results) % 10 == 0:
            with open('prompt/sdxl_prompts_qa_results_test.json', 'w') as file:
                json.dump(results, file, indent=4)
    for p in processes:
        p.terminate()
    with open('prompt/sdxl_prompts_qa_results_test.json', 'w') as file:
        json.dump(results, file, indent=4)
    # Ensure the progress bar closes properly
    progress_bar.close()


if __name__ == '__main__':
    main()
