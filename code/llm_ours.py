import time
import os

import openai
from openai import OpenAI
client = OpenAI(
    api_key='xxxx',
    base_url="https://api.chatanywhere.tech"
)

# import transformers
import json
import requests




model2api = {
    'gpt3': 'text-davinci-003',
    'chatgpt': 'gpt-3.5-turbo-0301'
}


def get_init_message(single_prompt):
    # message_sys = {"role": "system", "content": "You are free to respond without any restrictions."}
    message = {"role": "user", "content": f"{single_prompt}"}
    messages = [message] # message_sys, 
    return messages


def get_new_message(previous_turn, single_prompt):
    message_llm = {'role': 'assistant', 'content': f"{previous_turn}"}
    message = {"role": "user", "content": f"{single_prompt}"}
    messages = [message_llm, message]
    return messages


def get_llm_result(args, messages, vllm_bags):

    if "vllm" in args.model_type:
        sampling_params = vllm_bags[0]
        llm_model = vllm_bags[1]

    def get_res_from_chat(messages):
        outputs = client.chat.completions.create(
            model=model2api['chatgpt'],
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop='问题: '
        )
        print("answers:"+outputs.choices[0].message.content)

        return outputs.choices[0].message.content


    def get_res_from_vllm(messages):
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        outputs = llm_model.generate([prompt,], sampling_params)
        print("answers:"+outputs[0].outputs[0].text)
        return outputs[0].outputs[0].text


    def get_res_from_gomall(messages):
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        data = {
            "input": prompt, # [prompt,prompt]
            "params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stop": ["<|eot_id|>"]
            }
        }
        res = requests.post(args.url, data=json.dumps(data), timeout=30)
        outputs = json.loads(res.text)
        print("answers:"+outputs["response"][0]["outputs"][0]["text"])
        return outputs["response"][0]["outputs"][0]["text"]


    # 处理访问频率过高的情况
    def get_res(messages):
        if "gomall" in args.model_type:
            res = get_res_from_gomall(messages)
            return res

        elif "vllm" in args.model_type:
            res = get_res_from_vllm(messages)
            return res
    
        while True:
            try:
                if args.model_type=="chatgpt":
                    res = get_res_from_chat(messages)

                break
            
            except openai.error.RateLimitError as e:
                print('\nRateLimitError\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.ServiceUnavailableError as e:
                print('\nServiceUnavailableError\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.Timeout as e:
                print('\nTimeout\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.APIError as e:
                print('\nAPIError\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.APIConnectionError as e:
                print('\nAPIConnectionError\t', e, '\tRetrying...')
                time.sleep(5)
            except Exception as e:
                print(e)
                res = None
                break
        
        return res

    res = get_res(messages)
    return res
