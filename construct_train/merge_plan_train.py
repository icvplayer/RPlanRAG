import json
import random

def merge_and_shuffle_json_files(file1, file2, file3, file4, output_file):
   
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    with open(file3, 'r', encoding='utf-8') as f3:
        data3 = json.load(f3)

    with open(file4, 'r', encoding='utf-8') as f4:
        data4 = json.load(f4)
    
    
    merged_data = data1 + data2 + data3 + data4
    
   
    random.shuffle(merged_data)
    
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, ensure_ascii=False, indent=4)
    
    print(f"数据已合并并保存到 {output_file}")


file1 = 'training_summarize_data/Plan_training_data_2wikimqa.json'  
file2 = 'training_summarize_data/Plan_training_data_hotpotqa.json'  
file3 = 'training_summarize_data/Plan_training_data_musique.json'  
file4 = 'training_summarize_data/Plan_training_data_nq_10.json'  
output_file = 'training_summarize_data/Plan_training_data_all.json'  

merge_and_shuffle_json_files(file1, file2, file3, file4, output_file)