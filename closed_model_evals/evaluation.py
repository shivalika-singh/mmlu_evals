import pandas as pd
import argparse
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime
from datasets import Dataset
import json

def clean_answer(output):
    output_clean = output.encode('ascii', 'ignore').decode('ascii')
    return output_clean


def get_response_label(response,
                       choices, 
                       gold):
    """
    It parses the model's response and assigns the respective label.

    :param response: str, the long-form answer of the model
    :param gold: str, the gold label of the samples {A/B/C/D}
    return: str, the label of the model's response {A/B/C/D}
    """
    # change int to str if choices.keys are numbers
    # if isinstance(list(choices.keys())[0], int):
    #     choices = {str(x): y for x, y in choices.items()}

    response = clean_answer(response)
    # choices_reversed = {y: x for x, y in choices.items()}
    if response in choices:
        pred_label = response
    elif '<Answer>' in response:
        pred_label = response.split('<Answer>')[1].split('</Answer>')[0]
    else:
        res = response.split('\n')[0].strip()
        if res in choices:
            pred_label = res
        else:
            res = response.split(':')[0].strip()
            if res in choices:
                pred_label = res
            else:
                # return a random label other than the right one
                labels = ['A', 'B', 'C', 'D']
                labels.remove(gold)
                pred_label = random.choice(labels)
    
    return pred_label


def print_report(report):
    """
    It prints the performance scores for the running experiment.

    :param report: dict, the report with the performance scores
    """
    print('-'*35)
    print('\tPERFORMANCE REPORT')
    print('-'*35)
    print('Model'.ljust(20) + '{}'.format(report['Model']))
    print('Few-shot'.ljust(20) + '{}'.format(report['Few-shot']))
    # print('CoT'.ljust(20) + '{}'.format(report['CoT']))
    print('-'*35)
    print('Accuracy'.ljust(20) + '{}'.format(report['Accuracy']))
    print('F1'.ljust(20) + '{}'.format(report['F1']))
    print('-'*35)


def save_report(report):
    """
    It stores the evaluation results as a new entry.

    :param report: dict
    """
    file_name = 'outputs/reports.jsonl'
    reports = pd.read_json(file_name, lines=True)
    new_reports = pd.concat([reports, pd.DataFrame([report])])
    new_reports.to_json(file_name, orient='records')
    print("File: reports.json has been updated")


def main(model,
         cot,
         fs):
    """
    It runs the main evaluation pipeline for the given generations. 

    :param model: str, model name
    :param cot: bool
    :param fs: bool
    """
    # read prediction data
    cot_flag = '_cot' if cot else ''
    fs_flag = '_fs' if fs else ''
    file_name = "/home/shivalikasingh/mmlu_evals/aya-bench/closed_model_evals/outputs/mmlu_agnostic/fr/_claude-3-5-sonnet-20240620_4c76200c4278.json"
    #"./outputs/_gpt-4o-2024-08-06_524aa8e3d730.json"
    with open(file_name) as f:
        output_data = json.load(f)
    
    data = output_data['data']
    data = pd.DataFrame(data)
    
    # data = pd.read_json(file_name)
    print(data.columns)
    data['pred_label'] = data.apply(lambda x: get_response_label(response=x['output'], 
                                                                 choices=["A", "B", "C", "D"],
                                                                 gold=x['actual_answer']), axis=1)
    
    y_true = data['actual_answer'].values
    y_pred = data['pred_label'].values
    acc = accuracy_score(y_true=y_true,
                         y_pred=y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, 
                                                               y_pred=y_pred, 
                                                               average='weighted', 
                                                               zero_division=0)

    report = {'Model': model, 
              'Few-shot': 'Yes' if fs else 'No',
              #'CoT': 'Yes' if cot else 'No',
              'Accuracy': acc, 
              'Precision': precision,
              'Recall': recall,
              'F1': f1,
              'Timestamp': datetime.datetime.now()}

    # print results
    print_report(report)

    # save results
    save_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Name of the model used.',
                        default='Meta-Llama-3-8B')
    parser.add_argument('-c', '--cot',
                        help='Adding CoT to the prompt or not',
                        action='store_true',
                        default=None,
                        )
    parser.add_argument('-f', '--fs',
                        help='Adding few shot examples to the prompt or not',
                        action='store_true',
                        default="yes",
                        )
    
    args = parser.parse_args()

    main(model=args.model,
         cot=args.cot,
         fs=args.fs)