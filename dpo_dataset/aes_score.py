import tensorflow_hub as hub
import tensorflow as tf
import json
import os
import multiprocessing
import time ,tqdm
def load_image_bytes(image_path):
    """Load image and return its bytes."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return image_bytes
def process_prompt(task_queue, result_queue, device, vila_path):
    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)  # device is the GPU index as a string, e.g., '0' for the first GPU
    #加载模型
    model = hub.load(vila_path)

    predict_fn = model.signatures['serving_default']
    while not task_queue.empty():
        task,img_folder_path = task_queue.get()
        prompt_id = task["id"]
        task["images_aes_scores"]={}
        # 用于存储图片评分的字典
        for i in range(8):
            image_path = os.path.join(img_folder_path,f'{prompt_id}_{i}.png')
            image_bytes = load_image_bytes(image_path)
            predictions = predict_fn(tf.constant((image_bytes)))  # 注意：这里将字节数据放入一个列表中
            aesthetic_score = float(predictions['predictions'].numpy()[0][0])
            task["images_aes_scores"][f"{prompt_id}_{i}.png"] = aesthetic_score*100
        result_queue.put(task)

def main():
    vila_path = '../../models/vila_model'
    img_folder_path = 'sdxl_images'
    devices = ",".join(["0,1,2,3,4,5,6,7"]*1)
    prompt_path = './prompts/sdxl_prompts_vqa_clip_scores_results.json'
    save_file = 'prompts/sdxl_prompts_all_scores_results.json'
    
    with open(prompt_path, 'r') as file:
        data = json.load(file)
    
    task_queue = multiprocessing.Queue(maxsize=len(data))
    result_queue = multiprocessing.Queue(maxsize=len(data))

    for item in data:
        task_queue.put((item,img_folder_path))
    
    processes = []

    for device in devices.split(","):
        p = multiprocessing.Process(target=process_prompt, args=(task_queue, result_queue, int(device), vila_path))
        processes.append(p)
        p.start()
    
    # Initialize tqdm progress bar
    progress_bar = tqdm.tqdm(total=len(data))
    results = []
    while len(results)!=len(data):
        # print(f"\r {len(results)}/{len(data)}")
        while not result_queue.empty():
            item= result_queue.get()
            results.append(item)
            # Update progress bar
            progress_bar.update(1)
        time.sleep(1)
        if len(results)%10==0:
            with open(save_file, 'w') as file:
                json.dump(results, file, indent=4)
    for p in processes:
        p.terminate()
    results = sorted(results, key=lambda d: d['id'])
    with open(save_file, 'w') as file:
        json.dump(results, file, indent=4)
    # Ensure the progress bar closes properly
    progress_bar.close()
if __name__ == '__main__':
    main()