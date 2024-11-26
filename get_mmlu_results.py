import os
import sys
import json
import pandas as pd

def save_task_results(task, dir_name):
    results_df = pd.DataFrame(columns=['model', 'language', 'accuracy'])
    save_filename = f"{task}_results_check"
    for language in os.listdir(os.path.join(dir_name, task)):
        print("language:", language)
        results_dir = os.path.join(dir_name, task, language)
        for model in os.listdir(results_dir):
            print("model:", model)
            if model!=".ipynb_checkpoints":
                result_dir = os.path.join(dir_name, task, language, model)
                result_file = sorted([file for file in os.listdir(result_dir) if file.startswith("results")])
                num_result_files = len(result_file)
                result_file = os.path.join(result_dir, f'{result_file[0]}')
                with open(result_file, 'r') as f:
                    result_json = json.load(f)
                    acc_value = round(result_json['results'][f'{task}_{language.replace(".json","")}']['acc,none']*100, 2)
                    print("acc_value:", acc_value)
                    
                    new_row_df = pd.DataFrame([{'model': model, 'language': language.replace(".json",""), 'accuracy': acc_value, 'num_result_files': num_result_files}])
                    results_df = pd.concat([results_df, new_row_df])
    results_df.to_csv(f"{dir_name}/{save_filename}.csv")

    
if __name__ == "__main__":
    dir_name = "../results"
    
    # tasks = os.listdir(dir_name)
    tasks = ["mmlu_CS", "mmlu_CA"] 
    
    for task in tasks:
        print("task:", task)
        save_task_results(task, dir_name)