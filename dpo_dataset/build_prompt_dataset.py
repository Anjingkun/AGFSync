import openai
import json
import argparse
# from rouge_score import rouge_scorer
from tqdm import tqdm
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed

tokens_num = 0
with open('./prompt_12/system_prompt.txt','r') as file:
    system_prompt = file.read().strip()

tasks = [
            ("Natural Landscapes: Includes terrain, bodies of water, weather phenomena, and natural scenes.", "natural_landscapes.txt"),
            ("Cities and Architecture: Shows city skylines, iconic buildings, historical buildings, and architectural styles.", "city_architecture.txt"),
            ("People: Portrait photography, people's daily lives, and activities in specific environments.", "people_portraits.txt"),
            ("Animals: Photos of wild animals, pets, animals in zoos or aquariums, and their behaviors.", "animals_wildlife.txt"),
            ("Plants: Horticultural photography, wild plants, trees, flowers, and plant communities.", "plants_gardens.txt"),
            ("Food and Beverages: Culinary arts, gourmet photography, food preparation, traditional foods, drinks, and table settings.", "food_beverages.txt"),
            ("Sports and Fitness: Various sports, gym training, outdoor sports, and extreme sports.", "sports_fitness.txt"),
            ("Art and Culture: Paintings, sculptures, dance, concerts, theater performances, cultural festivals.", "art_culture.txt"),
            ("Technology and Industry: High-tech products, industrial facilities, and technology trends.", "tech_industry.txt"),
            ("Everyday Objects: Household items, office supplies, personal care products, furniture, decorative items.", "everyday_objects.txt"),
            ("Transportation: Cars, airplanes, trains, boats, bicycles, and their usage in different environments.", "transportation_modes.txt"),
            ("Abstract and Conceptual: Abstract art works, conceptual photography, creative expressions.", "abstract_conceptual.txt")
        ]

def configure_openai():
    openai.api_type = "azure"
    openai.api_base = ""
    openai.api_version = ""
    openai.api_key = ""


@retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
async def create_prompt_in_task_async(task_user_prompt:str, num: int, progress):
    global tokens_num
    # 替换{num}为您想要的数字，例如'10'
    new_system_prompt = system_prompt.replace('{num}', f'{num}')
    response = await openai.ChatCompletion.acreate(
        engine="gpt3_5_16k",
        messages=[
            {
                "role": "system",
                "content": new_system_prompt,
            },
            {
                "role": "user",
                "content": task_user_prompt,
            },
        ],
        temperature=0.9,
        max_tokens=4096,
        top_p=0.7,
        stop=None
    )
    progress.update()
    tokens_num += response.usage['total_tokens']
    try:
        msg = response.choices[0]["message"]["content"]
        result = json.loads((msg))["descriptions"]
    except:
        print("failed    " + msg)
        result = []
    return result

def load_prompts():
    with open("./prompt/twelve_categories_prompts.json", "r") as file:
        try:
            prompts = json.load(file)
        except:
            prompts = {}
    return prompts

def save_prompts(prompts):
    with open("./prompt/twelve_categories_prompts.json", "w") as file:
        json.dump(prompts, file)


async def process_task(task,txt_name, args, prompts):
    global tokens_num
    progress = tqdm(total=args.num_query_per_task, desc=str(tokens_num))
    with open(f'./prompt_12/{txt_name}') as file:
        task_user_prompt = file.read().strip()
    descriptions = await asyncio.gather(*(create_prompt_in_task_async(task_user_prompt, args.num_desc_per_query, progress) for _ in range(args.num_query_per_task)))
    num_uniq = 0
    all_desc = []
    for desc_list in descriptions:
        all_desc += desc_list
    for description in all_desc:
        num_uniq += 1
        prompts[task].append(description)
    
async def start_gen(args):
     # Load tasks
    prompts = load_prompts()
    for task,txt_name in tasks:
        if task not in prompts:
            prompts[task] = []
        await process_task(task,txt_name, args, prompts)
        save_prompts(prompts)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--num_query_per_task", type=int, default=10)
    parser.add_argument("--num_desc_per_query", type=int, default=20)
    args = parser.parse_args()
    configure_openai()
    for _ in range(args.num_runs):
        asyncio.run(start_gen(args))
