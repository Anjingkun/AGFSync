import json

num_of_gemini_key = 4
all_results = []
for i in range(num_of_gemini_key):
    with open(f"./processed_results_{i}.json", "r") as file:
        prompts = json.load(file)
    all_results.extend(prompts)
results = []
for result in all_results:
    if len(result["qa_pairs"]) >= 5:
        results.append(result)
for index, result in enumerate(results):
    result["id"] = index
results = sorted(results, key=lambda x: x['id'])
with open("../prompt/prompts_qas.json", "w") as file:
    json.dump(results, file, indent=4)
