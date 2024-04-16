import json
with open("sdxl_prompts_qa_results.json","r") as file:
    a = json.load(file)
for prompt in a:
    prompt["images_vqa_scores"]={}
    question_count = len(prompt["qa_pairs"])
    images_yes_count = {
        f"{prompt['id']}_0.png":0,
        f"{prompt['id']}_1.png":0,
        f"{prompt['id']}_2.png":0,
        f"{prompt['id']}_3.png":0,
        f"{prompt['id']}_4.png":0,
        f"{prompt['id']}_5.png":0,
        f"{prompt['id']}_6.png":0,
        f"{prompt['id']}_7.png":0,
    }
    for qa_pair in prompt["qa_pairs"]:
        for image_name, image_answer in qa_pair["images_answers"].items():
            if 'no' in image_answer.lower() or 'b' in image_answer.lower():
                continue
            else:
                images_yes_count[image_name]+=1
    for image_name, yes_count in images_yes_count.items():
        vqa_score = yes_count/question_count*100
        prompt["images_vqa_scores"][image_name]=vqa_score
with open("sdxl_prompts_vqa_score.json","w") as file:
    json.dump(a, file, indent=4)