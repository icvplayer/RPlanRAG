import json
import random
import re

def merge_and_shuffle_json_files(file1, file2, file3, file4, output_file):


    
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    with open(file3, 'r', encoding='utf-8') as f3:
        data3 = json.load(f3)

    with open(file4, 'r', encoding='utf-8') as f4:
        data4 = json.load(f4)
    
    
    merged_data = data1 + data2 + data3 +data4
    
    pattern = r'\n\nThe answer to this question:\n[^\n]+'
    for one_data in merged_data:
        replaced_str = re.sub(pattern, '', one_data["instruction"])
        replaced_str = replaced_str.replace("You will be given a context (The data tables that contain relationships and sources) , a question and the answer to this question", "You will be given a context (The data tables that contain relationships and sources) and a question")
        # replace_str = re.sub(pattern, need_replace_str, one_data["instruction"], flags=re.DOTALL)
        one_data["instruction"] = replaced_str
        marker="<|end|>"
        idx = one_data["output"].rfind(marker)
        if idx != -1:
            one_data["output"] = one_data["output"][:idx + len(marker)]

        remove_example_pattern = r'#####Here are two examples: #####.*?#####Example 2: #####'
        replacement = '#####Here is an example: #####'
        one_data["instruction"] = re.sub(remove_example_pattern, replacement, one_data["instruction"], flags=re.DOTALL)



   
    random.shuffle(merged_data)
    
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, ensure_ascii=False, indent=4)
    
    print(f"数据已合并并保存到 {output_file}")


file1 = 'training_summarize_data/summary_training_data_nq_10.json' 
file2 = 'training_summarize_data/summary_training_data_2wikimqa.json'  
file3 = 'training_summarize_data/summary_training_data_hotpotqa.json'  
file4 = 'training_summarize_data/summary_training_data_musique.json' 
output_file = 'training_summarize_data/Summary_training_data_all.json' 

merge_and_shuffle_json_files(file1, file2, file3, file4,  output_file)