import re
import numpy as np

def metric(y_true, y_pred):
    return np.mean((np.array(y_true)-np.array(y_pred))**2)

def get_resul_deepseek(text):
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
        # 修改这里的正则表达式，支持负数
        match = re.search(r'(-?\d+\.\d+|-?\d+)$', item)
        if match:
            number = float(match.group(0))
            res_list.append(number)    
    return res_list


def extract_values(text):
    pattern = r'(-?\d+\.\d+)'  # 匹配带正负号的浮点数
    values = re.findall(pattern, text)
    return [float(value) for value in values]


def MSE(ground_truth,answer):
    ground_truth = extract_values(ground_truth)
    answer = get_resul_deepseek(answer)

    if len(answer) >= len(ground_truth):
        return 1-metric(ground_truth, answer[:len(ground_truth)])/100
    else:
        return 0
