# -- coding: utf-8 --
import json
import google.generativeai as genai
import concurrent.futures
import sys
import os
# os.environ['https_proxy'] = 'http://127.0.0.1:7997'
# os.environ['http_proxy'] = 'http://127.0.0.1:7997'
# 获取命令行参数
if len(sys.argv) != 2:
    print("Usage: python script.py <key_index>")
    sys.exit(1)

key_index = int(sys.argv[1])  # 将命令行参数转换为整数
# 定义API密钥列表
api_keys = [""]
# 设置当前脚本应使用的API密钥的索引（0到4之间）
if key_index < 0 or key_index >= len(api_keys):
    print("Invalid key index. Must be between 0 and", len(api_keys) - 1)
    sys.exit(1)

# 配置API密钥
genai.configure(api_key=api_keys[key_index])

def process_prompt(item, system_prompt):
    """处理单个prompt，获取6次有效响应"""
    model = genai.GenerativeModel('gemini-pro')
    user_prompt = item['content']
    prompt_id = item['id']

    combined_prompt = [{"role": "system", "content": system_prompt}, {"role":
"user", "content": user_prompt}]
    combined_prompt = str(combined_prompt)
    responses_for_prompt = []
    success_count = 0
    while success_count < 6:  # 确保每个 prompt 得到6次有效响应
        try:
            response = model.generate_content(combined_prompt).text

            # 尝试解析 response JSON 数据
            response_content = json.loads(response)
            
            responses_for_prompt.extend(response_content)  # 添加到响应数组
            success_count += 1
        except Exception as e:
            # 如果解析失败，继续尝试直到成功为止
            continue
    print(prompt_id)

    return {
        "id": prompt_id,
        "prompt": user_prompt,
        "qa_pairs": responses_for_prompt
    }

# 读取系统提示
with open('./prompts/system_prompt.txt', 'r') as file:
    system_prompt = file.read().strip()
# 使用多线程处理每个prompt
try:
    with open(f'./twelve/intermediate_qa_pairs_gemini_{key_index}.json', 'r') as file:
        all_responses = json.load(file)
except:
    all_responses = []
id_list = [item['id'] for item in all_responses]
# 读取并分割用户提示
with open('../prompt/prompts_filtered_90.json', 'r') as file:
    all_prompts = json.load(file)
    total_prompts = len(all_prompts)
    chunk_size = total_prompts // len(api_keys)
    remainder = total_prompts % len(api_keys)

    if key_index < len(api_keys) - 1:
        start_index = key_index * chunk_size
        end_index = start_index + chunk_size
    else:
        # 最后一个API密钥处理剩余的所有提示
        start_index = key_index * chunk_size
        end_index = total_prompts  # 包括余数部分

    all_prompts_with_images = all_prompts[start_index:end_index]
    prompts_with_images = []
    for item in all_prompts_with_images:
        if item['id'] not in id_list:
            prompts_with_images.append(item)
        else:
            print(f"skip:{item['id']}")


with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process_prompt, item, system_prompt) for item in prompts_with_images]
    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
        all_responses.append(future.result())
        if i % 10 == 0:  # 每10个prompts保存一次
            with open(f'./twelve/intermediate_qa_pairs_gemini_{key_index}.json', 'w') as intermediate_file:
                json.dump(all_responses, intermediate_file, ensure_ascii=False, indent=4)
# 将结果保存到JSON文件
output_file = f'./twelve/processed_results_{key_index}.json'
with open(output_file, 'w') as file:
    json.dump(all_responses, file, ensure_ascii=False, indent=4)
