import numpy as np
from openai import OpenAI
import re


def get_result(text):
    res_list = []
    pattern = r"```*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match is None:
        pattern = r"\|.*\|"
        text_list = re.findall(pattern, text)
    else:
        text = match.group(1).strip()
        text_list = text.split('\n')

    for index, item in enumerate(text_list):
        if index == 0:
            continue
        item = item.strip()
        if item.endswith('|'):
            item = item.rstrip('|').strip()
        match = re.search(r'(-?\d+\.\d+|-?\d+)$', item)
        if match:
            number = float(match.group(0))
            res_list.append(number)
    return res_list


def getQwenResult(model_name, content, top_p=0.7):
    client = OpenAI(
        base_url="YOUR_BASE_URL",
        api_key="YOUR_API_KEY",
    )
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": content
        }],
        top_p=top_p,
        stream=True,
        max_tokens=7000,
        timeout=600,
    )
    
    answer = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            content_chunk = chunk.choices[0].delta.content
            answer += content_chunk
            if "</answer>" in answer:
                end_pos = answer.find("</answer>") + len("</answer>")
                answer = answer[:end_pos]
                break
    
    return answer


def call_GPT(content, model_name="gpt-4.1", top_p=0.4):
    client = OpenAI(
        base_url="YOUR_BASE_URL",
        api_key="YOUR_API_KEY",
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": content
        }],
        top_p=top_p,
        stream=False,
        max_tokens=4000,
    )
    answer = completion.choices[0].message.content
    return answer


def call_R1(content, api_key='YOUR_API_KEY'):
    client = OpenAI(base_url="YOUR_BASE_URL", api_key=api_key)
    response = client.chat.completions.create(
        model="YOUR_MODEL_NAME",
        messages=[
            {"role": "user", "content": content}, ], stream=False,
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096, )

    reasoning = response.choices[0].message.model_extra['reasoning_content']
    answer = response.choices[0].message.content
    return "<think>" + reasoning + "</think>\n<answer>" + answer + "</answer>"


def MSE(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def MAE(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
