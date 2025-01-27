import argparse
import math
import json
import time
from pprint import pprint
from openai import AsyncOpenAI, AsyncAzureOpenAI, APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
import os
import pandas as pd
from tqdm import tqdm
import pathlib
import traceback
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import asyncio, dataclasses
from dotenv import load_dotenv
import logging, sys
from anthropic import (AsyncAnthropic, RateLimitError as AnthropicRateLimitError, APIConnectionError as AnthropicAPIConnectionError, APITimeoutError as AnthropicAPITimeoutError, InternalServerError as AnthropicInternalServerError)
from datasets import load_dataset

logging.basicConfig(stream=sys.stderr, level=logging.WARN)
logger = logging.getLogger(__name__)


from utils import read_json, write_json, generate_unique_id, batched


OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-0125-preview", "gpt-4o-2024-08-06"]
GOOGLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
API_MODELS = OPENAI_MODELS + GOOGLE_MODELS + ANTHROPIC_MODELS

SYSTEM_PROMPT="""
You are an AI assistant, and your role is to provide accurate answers to multiple-choice questions across various subjects and languages.

Each user prompt will present a series of questions with four options: A, B, C, and D. Your task is simple: identify the correct answer choice for each question and respond with only the letter corresponding to the correct option.

Do not provide any explanations or additional information. Your response should be concise and to the point, consisting solely of the letter representing the correct answer.

For example, if the correct answer is option B, your response should be:

B

Proceed to the next question once you have provided your answer, and continue this process until all questions have been addressed.
"""

mmlu_subjects_formatted_names_map={'business_ethics': 'Business Ethics',
 'econometrics': 'Econometrics',
 'global_facts': 'Global Facts',
 'high_school_european_history': 'High School European History',
 'high_school_geography': 'High School Geography',
 'high_school_government_and_politics': 'High School Government And Politics',
 'high_school_macroeconomics': 'High School Macroeconomics',
 'high_school_microeconomics': 'High School Microeconomics',
 'high_school_psychology': 'High School Psychology',
 'high_school_us_history': 'High School US History',
 'high_school_world_history': 'High School World History',
 'human_aging': 'Human Aging',
 'human_sexuality': 'Human Sexuality',
 'international_law': 'International Law',
 'jurisprudence': 'Jurisprudence',
 'logical_fallacies': 'Logical Fallacies',
 'management': 'Management',
 'marketing': 'Marketing',
 'miscellaneous': 'Miscellaneous',
 'moral_disputes': 'Moral Disputes',
 'nutrition': 'Nutrition',
 'philosophy': 'Philosophy',
 'prehistory': 'Prehistory',
 'professional_accounting': 'Professional Accounting',
 'professional_law': 'Professional Law',
 'professional_medicine': 'Professional Medicine',
 'professional_psychology': 'Professional Psychology',
 'public_relations': 'Public Relations',
 'security_studies': 'Security Studies',
 'sociology': 'Sociology',
 'us_foreign_policy': 'US Foreign Policy',
 'virology': 'Virology',
 'high_school_computer_science': 'High School Computer Science',
 'college_computer_science': 'College Computer Science',
 'computer_security': 'Computer Security',
 'astronomy': 'Astronomy',
 'abstract_algebra': 'Abstract Algebra',
 'college_chemistry': 'College Chemistry',
 'college_mathematics': 'College Mathematics',
 'electrical_engineering': 'Electrical Engineering',
 'elementary_mathematics': 'Elementary Mathematics',
 'formal_logic': 'Formal Logic',
 'high_school_chemistry': 'High School Chemistry',
 'high_school_mathematics': 'High School Mathematics',
 'high_school_physics': 'High School Physics',
 'high_school_statistics': 'High School Statistics',
 'machine_learning': 'Machine Learning',
 'medical_genetics': 'Medical Genetics',
 'college_physics': 'College Physics',
 'conceptual_physics': 'Conceptual Physics',
 'anatomy': 'Anatomy',
 'clinical_knowledge': 'Clinical Knowledge',
 'college_biology': 'College Biology',
 'college_medicine': 'College Medicine',
 'high_school_biology': 'High School Biology',
 'world_religions': 'World Religions',
 'moral_scenarios': 'Moral Scenarios'}

language_mapping = {
# 'hi': 'Hindi',
# 'ar': 'Arabic',
# 'fr': 'French',
# 'es': 'Spanish',
'sw': 'Swahili',
# 'bn': 'Bengali',
# 'de': 'German',
# 'id': 'Indonesian',
# 'it': 'Italian',
# 'ja': 'Japanese',
# 'ko': 'Korean',
# 'pt': 'Portuguese',
# 'zh': 'Chinese',
# 'yo': 'Yoruba',

    
# 'ru': 'Russian',
# 'vi': 'Vietnamese',
# 'ms': 'Malay',
# 'cs': 'Czech',
# 'tr': 'Turkish',
# 'pl': 'Polish',



# 'en': 'English',
#'ro': 'Romanian',
#'fil': 'Filipino',
#'sr': 'Serbian',
#'lt': 'Lithuanian',
#'so': 'Somali',

#pending below
# 'am': 'Amharic',
# 'el': 'Greek',
# 'en': 'English',
# 'fa': 'Persian',
# 'ha': 'Hausa',
# 'he': 'Hebrew',
# 'ig': 'Igbo',
# 'ky': 'Kyrgyz',
# 'mg': 'Malagasy',
# # 'ne': 'Nepali',
# # 'nl': 'Dutch',
# 'ny': 'Chichewa',
# 'si': 'Sinhala',
# 'sn': 'Shona',
# 'sv': 'Swedish',
# 'te': 'Telugu',
# 'uk': 'Ukrainian',

# 'zh-CN': 'Chinese',

# }

# SKIP_LANGUAGES = {
# 'ne': 'Nepali',
# 'ha': 'Hausa',
# 'sv': 'Swedish',
# 'el': 'Greek',
# 'nl': 'Dutch',
# 'ig': 'Igbo',
# 'fil': 'Filipino',
# 'sr': 'Serbian',
# 'lt': 'Lithuanian',
# 'pl': 'Polish',
# 'so': 'Somali',
# 'sn': 'Shona',
# 'he': 'Hebrew',
# 'ny': 'Chichewa',
# 'ky': 'Kyrgyz',
# 'mg': 'Malagasy',
}

with open("translated_prompts.json") as f:
    MMLU_SUBJECT_LANG_PROMPTS = json.load(f)


with open("translated_prompts_bn_sw.json") as f:
    MMLU_SUBJECT_LANG_PROMPTS_BN_SW = json.load(f)

@dataclasses.dataclass
class ModelResponse:
    text: str
    usage: dict = None
    exception: Exception = None


def get_openai_model_args(model_args):
    openai_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            openai_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            openai_model_args["max_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            openai_model_args["top_p"] = model_args["top_p"]
        if "frequency_penalty" in model_args:
            openai_model_args["frequency_penalty"] = model_args["frequency_penalty"]
        if "presence_penalty" in model_args:
            openai_model_args["presence_penalty"] = model_args["presence_penalty"]

    return openai_model_args


@retry(retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10), before_sleep=before_sleep_log(logger, logging.DEBUG))
async def openai_chat_completion(client, messages, model="gpt-3.5-turbo", model_args=None):
    openai_model_args = get_openai_model_args(model_args)
    text = ""
    exception = None
    # print("messages:", messages)
    response = await client.chat.completions.create(model=model, messages=messages, **openai_model_args)
    content = response.choices[0].message.content
    
    if content is None:
        exception = f"Finish reason: {response.choices[0].finish_reason}"
        usage = None
    else:
        text = content.strip()
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
    
    return ModelResponse(text, usage, exception)

async def evaluate_openai_model(client, model, user_prompt, system_prompt=None, model_args=None):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    
    messages.append({"role": "user", "content": user_prompt.strip()})

    return await openai_chat_completion(client, messages, model=model, model_args=model_args)


def get_anthropic_model_args(model_args):
    anthropic_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            anthropic_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            anthropic_model_args["max_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            anthropic_model_args["top_p"] = model_args["top_p"]
        if "top_k" in model_args and model_args["top_k"] is not None:
            anthropic_model_args["top_k"] = model_args["top_k"]

    return anthropic_model_args


@retry(retry=retry_if_exception_type((AnthropicAPITimeoutError, AnthropicAPIConnectionError, AnthropicRateLimitError, AnthropicInternalServerError)), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10), before_sleep=before_sleep_log(logger, logging.DEBUG))
async def anthropic_chat_completion(client, messages, system_prompt=None, model="claude-3-5-sonnet-20240620", model_args=None):
    anthropic_model_args = get_anthropic_model_args(model_args)
    exception = None

    try:
        response = await client.messages.create(model=model, messages=messages, system=system_prompt, **anthropic_model_args)
        text = response.content[0].text.strip()
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    except (AttributeError, ValueError) as e:
        text = ""
        usage = {}
        exception = e
    
    return ModelResponse(text, usage, exception)


async def evaluate_anthropic_model(client, model, user_prompt, system_prompt=None, model_args=None):
    messages = [{"role": "user", "content": user_prompt.strip()}]
    return await anthropic_chat_completion(client, messages, system_prompt=system_prompt, model=model, model_args=model_args)



async def evaluate_api_model(client, model, batch, model_args=None):
    tasks = []
    
    for sample in batch:
        if model in OPENAI_MODELS:
            tasks.append(asyncio.create_task(evaluate_openai_model(client, model, sample['user_prompt'], SYSTEM_PROMPT, model_args=model_args)))
        elif model in ANTHROPIC_MODELS:
            tasks.append(asyncio.create_task(evaluate_anthropic_model(client, model, sample['user_prompt'], SYSTEM_PROMPT, model_args=model_args)))
        else:
            raise ValueError(f"Model {model} not supported")
    
    results = await asyncio.gather(*tasks)

    return results


def configure_openai_client(api_key, is_openai_azure=False):
    if is_openai_azure:
        endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT", "https://sigturk-openai.openai.azure.com/")
        client = AsyncAzureOpenAI(
            api_key = api_key if api_key is not None else os.getenv("AZURE_OPENAI_API_KEY"),
            api_version = '2024-02-15-preview',
            azure_endpoint=endpoint
        )
    else:
        client = AsyncOpenAI(api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"))
    
    return client


def configure_anthropic_client(api_key):
    return AsyncAnthropic(api_key=api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY"))


def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)


def _write_error(error_path, sample, exception):
    with open(error_path, "a") as error_file:
        error_file.write(f"Error for sample {sample['user_prompt']}: {str(exception)}\n")
        error = "".join(traceback.format_exception(type(exception), value=exception, tb=exception.__traceback__))
        error_file.write(error)
        error_file.write("\n")

def prepare_user_prompt(x, language, dev_df):
    subject = x['subject']
    option_0 = x['option_0']
    option_1 = x['option_1']
    option_2 = x['option_2']
    option_3 = x['option_3']
    question = x['question']
    
    if subject=="college_mathematics_test.csv_YO-NG.csv":
        subject = "college_mathematics"
    if subject=="security_studies_test-sw-KE.csv":
        subject = "security_studies"
    if subject=="college_mathematics_test.csv_sw-KE.csv": 
        subject="college_mathematics"

    print("subject:", subject)
    print("language:", language)
    
    subject_df = dev_df[dev_df['subject']==subject]

    if language!="en":
        if language =='bn' or language == 'sw':
            # print("reading bengali or swahili prompts")
            IN_LANGUAGE_PROMPT = MMLU_SUBJECT_LANG_PROMPTS_BN_SW[language][subject]
        else:
            IN_LANGUAGE_PROMPT = MMLU_SUBJECT_LANG_PROMPTS[language][subject]
    else:
        subject_name = mmlu_subjects_formatted_names_map[subject]
        IN_LANGUAGE_PROMPT = f"The following are multiple choice questions (with answers) for {subject_name}."
    
    user_prompt = f""""{IN_LANGUAGE_PROMPT}
    
    Question: {subject_df.iloc[0]['question']}
    
    Options:
    A. {subject_df.iloc[0]['option_0']}
    B. {subject_df.iloc[0]['option_1']}
    C. {subject_df.iloc[0]['option_2']}
    D. {subject_df.iloc[0]['option_3']}
    
    Answer: {subject_df.iloc[0]['answer']}
    
    Question: {subject_df.iloc[1]['question']}
    
    Options:
    A. {subject_df.iloc[1]['option_0']}
    B. {subject_df.iloc[1]['option_1']}
    C. {subject_df.iloc[1]['option_2']}
    D. {subject_df.iloc[1]['option_3']}
    
    Answer: {subject_df.iloc[1]['answer']}
    
    Question: {subject_df.iloc[2]['question']}
    
    Options:
    A. {subject_df.iloc[2]['option_0']}
    B. {subject_df.iloc[2]['option_1']}
    C. {subject_df.iloc[2]['option_2']}
    D. {subject_df.iloc[2]['option_3']}
    
    Answer: {subject_df.iloc[2]['answer']}
    
    Question: {subject_df.iloc[3]['question']}
    
    Options:
    A. {subject_df.iloc[3]['option_0']}
    B. {subject_df.iloc[3]['option_1']}
    C. {subject_df.iloc[3]['option_2']}
    D. {subject_df.iloc[3]['option_3']}
    
    Answer: {subject_df.iloc[3]['answer']}
    
    Question: {subject_df.iloc[4]['question']}
    
    Options:
    A. {subject_df.iloc[4]['option_0']}
    B. {subject_df.iloc[4]['option_1']}
    C. {subject_df.iloc[4]['option_2']}
    D. {subject_df.iloc[4]['option_3']}
    
    Answer: {subject_df.iloc[4]['answer']}
    
    Question: {question}
    
    Options:
    A. {option_0}
    B. {option_1}
    C. {option_2}
    D. {option_3}
    
    Answer:
    """
    return user_prompt
    

async def main():
    load_dotenv() 

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, help="Path to evaluation data in json", default="")
    parser.add_argument("-a", "--api-key", type=str, help="Model API Key")
    parser.add_argument("-oa", "--openai-azure", action="store_true", help="If OpenAI on Azure")
    parser.add_argument("-m", "--model", type=str, help="Model to use for evaluation", default="claude-3-5-sonnet-20240620") #
    parser.add_argument("-t", "--temperature", type=float, help="Temperature for generation", default=0.0)
    parser.add_argument("-g", "--max-tokens", type=none_or_int, help="Max tokens for generation", default=2048) #512)
    parser.add_argument("-p", "--top-p", type=float, help="Top-p for generation", default=1)
    parser.add_argument("-k", "--top-k", type=float, help="Top-k for generation", default=None)
    parser.add_argument("-fp", "--frequency-penalty", type=float, help="Frequency penalty for generation", default=0)
    parser.add_argument("-pp", "--presence-penalty", type=float, help="Presence penalty for generation", default=0)
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory for evaluation results", default="outputs")
    parser.add_argument("-c", "--cache-dir", type=str, help="Cache directory for model", default="~/.cache")
    parser.add_argument("-mp", "--model-path", type=str, help="Model path to use for evaluation", default=None)
    parser.add_argument("-tp", "--tokenizer-path", type=str, help="Tokenizer path to use for evaluation", default=None)
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size for evaluation", default=100)
    
    args = parser.parse_args()
    client = None

    if args.model in API_MODELS:
        if args.model in OPENAI_MODELS:
            client = configure_openai_client(args.api_key, args.openai_azure)
            print("configured openai client")
        elif args.model in ANTHROPIC_MODELS:
            client = configure_anthropic_client(args.api_key)
            print("configured anthropic client")
    

    datasets = ['mmlu_CS', 'mmlu_CA'] #'mmlu_subset_translated'] #'mmlu_CS_subset'] # 'mmlu_agnostic_human_edited', 'mmlu_subset_translated', 

    for ds_name in datasets:
        if ds_name == 'mmlu_CA':
           required_languages = ['sw'] #, ] #'hi' #'ar', 'es', 'fr',, 'bn'
        else:
           required_languages = language_mapping.keys()

        for language in required_languages:
            try:
                if ds_name =="mmlu_agnostic_human_edited" and language in ['hi', 'ar', 'fr', 'es', 'ru', 'de', 'id', 'it', 'ja', 'ko', 'pt', 'zh', 'vi']:
                    print("skipping language:", language, "for dataset:", ds_name)
                    continue
                if ds_name == "mmlu_subset_translated" and language in ['de', 'es', 'id', 'it', 'ja', 'ko', 'nl','pl', 'ru', 'sn', 'so', 'sv', 'yo', 'zh']:
                    print("skipping language:", language, "for dataset:", ds_name)
                    continue
                if ds_name == "mmlu_full_gtranslate" and language in ['hi']:
                    print("skipping language:", language, "for dataset:", ds_name)
                    continue
                print(f"processing {language} for {ds_name}")
                config = language
                if language=="zh" and ds_name=="mmlu_subset_translated":
                    config="zh-CN"
            
                mmlu = load_dataset(f"CohereForAI/{ds_name}",config)
                mmlu_ds_dev = mmlu['dev']
                mmlu_ds_dev.set_format("pandas")
                dev_df = mmlu_ds_dev[:]

                # small_dataset = mmlu_ds_test.select([0, 50]) # 600, 1460, 2000
                # small_dataset.set_format("pandas")
                # test_df = small_dataset[:]
                
                mmlu_ds_test = mmlu['test']
                mmlu_ds_test.set_format("pandas")
                test_df = mmlu_ds_test[:]

                test_df['user_prompt'] = test_df.apply(lambda x: prepare_user_prompt(x, language, dev_df), axis=1)
                # print(test_df)

                data = []
                for idx, row in test_df.iterrows():
                    data.append({'user_prompt': row['user_prompt'],
                                'actual_answer': row['answer'],
                                'question': row['question'],
                                'option_0': row['option_0'],
                                'option_1': row['option_1'],
                                'option_2': row['option_2'],
                                'option_3': row['option_3'],
                                'subject': row['subject'],
                                'language': language,
                                # 'sample_id': row['sample_id'],
                                'required_sample': row['required_sample'],
                                'culture_label': row['culture_label'],
                                'old_culture_label': row['old_culture_label'],
                                'index_id': row['index_id']
                                })

                # data = pd.read_json(args.datapath).to_dict('records')
                print('Data size: {}'.format(len(data)))
                
                outputs = {
                    "metadata": {
                        "source": args.datapath,
                        "size": len(data),
                        "model": args.model,
                        "model_path": args.model_path,
                        "tokenizer_path": args.tokenizer_path,
                        "cache_dir": args.cache_dir,
                        "batch_size": args.batch_size,
                        "openai_azure": args.openai_azure,
                        "model_args": {
                            "temperature": args.temperature,
                            "max_tokens": args.max_tokens,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                            "frequency_penalty": args.frequency_penalty,
                            "presence_penalty": args.presence_penalty
                        }
                    },
                    "metrics": {},
                    "data": data
                }

                pathlib.Path(os.path.join(args.output_dir, ds_name, language)).mkdir(parents=True, exist_ok=True)
                datapath = pathlib.Path(args.datapath)
                unique_id = generate_unique_id()
                output_path = os.path.join(args.output_dir, ds_name, language, f"{datapath.stem}_{args.model}_{unique_id}.json")
                error_path = os.path.join(args.output_dir, ds_name, language, f"{datapath.stem}_{args.model}_{unique_id}_errors.txt")

                print(f"Writing to {output_path}")

                model_args = {
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "frequency_penalty": args.frequency_penalty,
                    "presence_penalty": args.presence_penalty
                }
                # pprint(model_args)

                for batch in tqdm(batched(data, size=args.batch_size), total=math.ceil(len(data)/args.batch_size)):
                    # print(batch)
                    results = []
                    results = await evaluate_api_model(client, args.model, batch, model_args)
                    # print("results:", results)
                    try:
                        for sample, result in zip(batch, results):
                            sample["output"] = result.text
                            sample["usage"] = result.usage
                            sample["result_id"] = generate_unique_id()
                            if result.exception is not None:
                                sample["exception"] = str(result.exception)
                        
                        write_json(outputs, output_path, ensure_ascii=False)
                    except Exception as e:
                        _write_error(error_path, sample, e)

                write_json(outputs, output_path, ensure_ascii=False)
                
                time.sleep(60)
            except Exception as e:
                traceback.print_exc()
                print(f"processing FAILED for {language} for {ds_name}")
            # break
        # break
        


if __name__ == "__main__":
    asyncio.run(main())
