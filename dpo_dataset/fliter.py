import json
from FlagEmbedding import FlagModel
import torch

tasks = [
    ("Natural Landscapes: Includes terrain, bodies of water, weather phenomena, and natural scenes.",
     "natural_landscapes.txt"),
    ("Cities and Architecture: Shows city skylines, iconic buildings, historical buildings, and architectural styles.",
     "city_architecture.txt"),
    ("People: Portrait photography, people's daily lives, and activities in specific environments.",
     "people_portraits.txt"),
    ("Animals: Photos of wild animals, pets, animals in zoos or aquariums, and their behaviors.",
     "animals_wildlife.txt"),
    ("Plants: Horticultural photography, wild plants, trees, flowers, and plant communities.", "plants_gardens.txt"),
    (
        "Food and Beverages: Culinary arts, gourmet photography, food preparation, traditional foods, drinks, and table settings.",
        "food_beverages.txt"),
    ("Sports and Fitness: Various sports, gym training, outdoor sports, and extreme sports.", "sports_fitness.txt"),
    ("Art and Culture: Paintings, sculptures, dance, concerts, theater performances, cultural festivals.",
     "art_culture.txt"),
    ("Technology and Industry: High-tech products, industrial facilities, and technology trends.", "tech_industry.txt"),
    ("Everyday Objects: Household items, office supplies, personal care products, furniture, decorative items.",
     "everyday_objects.txt"),
    ("Transportation: Cars, airplanes, trains, boats, bicycles, and their usage in different environments.",
     "transportation_modes.txt"),
    ("Abstract and Conceptual: Abstract art works, conceptual photography, creative expressions.",
     "abstract_conceptual.txt")
]
model = FlagModel('BAAI/bge-large-en-v1.5',
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)
similarity_threshold = 0.90
with open('./prompt/twelve_categories_prompts.json', 'r') as file:
    tasks_prompts = json.load(file)
for task, _ in tasks:
    task_prompts = tasks_prompts[task]
    embeddings = model.encode(task_prompts)
    similarity_matrix = embeddings @ embeddings.T
    # 过滤重复项
    filtered_responses = []
    seen = set()

    for i, task_prompt in enumerate(task_prompts):
        if i in seen:
            continue

        for j in range(i + 1, len(task_prompts)):
            if similarity_matrix[i, j] >= similarity_threshold:
                seen.add(j)

        filtered_responses.append(task_prompt)

    tasks_prompts[task] = filtered_responses
all_results = []
count = 0
for task, responses in tasks_prompts.items():
    for prompt in responses:
        all_results.append({
            "id": count,
            "content": prompt
        })
        count += 1

with open('./prompt/prompts_filtered_90.json', 'w') as file:
    json.dump(all_results, file, indent=4)
