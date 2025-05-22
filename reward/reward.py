import string
import re
import numpy as np
import torch
import torch.nn as nn


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def decompose(x):
    x = np.array(x)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.unsqueeze(2)
    model = series_decomp(kernel_size=25)
    season, trend = model(x)

    season = season.squeeze(0)
    season = season.squeeze(1)
    trend = trend.squeeze(0)
    trend = trend.squeeze(1)
    season = season.detach().numpy()
    trend = trend.detach().numpy()
    return season, trend

def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


# 季节性和趋势的均方误差
def mean_squared_error_season_trend(y_true, y_pred):
    season_true, trend_true = decompose(y_true)
    season_pred, trend_pred = decompose(y_pred)
    return np.mean((np.array(season_true) - np.array(season_pred)) ** 2), np.mean(
        (np.array(trend_true) - np.array(trend_pred)) ** 2),


# 从string中提取答案
def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1) if match else text


def extract_qwen_results(text):
    text = extract_answer(text)
    res_list = []
    # 尝试匹配代码块
    pattern = r"```*([\s\S]*?)```"
    match = re.search(pattern, text)
    # 如果没有匹配到代码块，使用表格格式匹配
    text_list = re.findall(r"\|.*\|", text) if match is None else match.group(1).strip().split('\n')
    for index, item in enumerate(text_list):
        # 跳过表头
        if index == 0:
            continue
        item = item.strip().rstrip('|').strip()
        # 匹配数值
        num_match = re.search(r'(-?\d+\.\d+|-?\d+)$', item)
        if num_match:
            res_list.append(float(num_match.group(0)))
    return res_list


# 从string gt 提取数据
def extract_ground_truth_values(text):
    pattern = r'(-?\d+\.\d+)'
    return [float(value) for value in re.findall(pattern, text)]


# 规范化答案文本
def normalize_answer(text):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


# 提取解决方案中的答案
def extract_solution_answer(solution_str):
    match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    return match.group(1).strip() if match else None


# 计算格式得分
def compute_format_score(solution_str):
    if solution_str is None:
        return -1.0
    try:
        # 检查是否包含 <think> 和 <answer> 标签
        think_answer_match = re.search(r'<think>(.*?)</think>\n<answer>(.*?)</answer>', solution_str, re.DOTALL)
        return 0.0 if think_answer_match else -1.0
    except Exception as e:
        print(f"[DEBUG] Error in compute_format_score: {e}")
        return -1.0


# 计算答案正确性奖励
def compute_answer_score(solution_str, ground_truth):
    ground_truth = extract_ground_truth_values(ground_truth)
    answer = extract_qwen_results(solution_str)
    if len(answer) >= len(ground_truth):
        norm_answer, norm_gt = reworad_norm(answer, ground_truth)
        mse = mean_squared_error(norm_gt, norm_answer[:len(ground_truth)])
        score = (1 - 1 / (1 + np.exp(-2 * mse))) * 2
        return score * 0.6
    else:
        return 0


# 根据长度给予适当奖励（目前上限96/96），未来考虑让LLM输出尽可能长的输出
def compute_score_length(solution_str, ground_truth):
    ground_truth = extract_ground_truth_values(ground_truth)
    answer = extract_qwen_results(solution_str)
    if len(answer) >= len(ground_truth):
        return 0.1
    else:
        # 一旦模型能够输出一定长度的时序数据，即使长度不足96
        # 也根据长度进行适度奖励
        # 防止一直给予负奖励训练失败
        return 0.1 * len(answer) / len(ground_truth)
    

def reworad_norm(x_list, y_list):
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    mu = np.nanmean(x_array)  # 使用忽略NaN的均值
    denominator = np.nanmax(np.abs(x_array - mu))  # 忽略NaN的最大值
    denominator = max(denominator, 1e-8)  # 添加保护值
    x_result = (x_array - mu) / denominator
    y_result = (y_array - mu) / denominator
    return x_result.tolist(), y_result.tolist()



    
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    score = compute_format_score(solution_str)
    score += compute_score_length(solution_str, ground_truth)
    score += compute_answer_score(solution_str, ground_truth)
    score += conmute_change_point(solution_str, ground_truth)
    return score



def change_point(data):
    #data=data.tolist()
    change_point_max=[]
    change_point_min=[]
    data_mirror_forward=[data[2],data[1]]

    data_mirror_backword=[data[-2],data[-1]]
    data_new=data_mirror_forward+data+data_mirror_backword

    for i in range(len(data)):
        if data_new[i+2]>=data_new[i] and data_new[i+2]>=data_new[i+1] and data_new[i+2]>=data_new[i+3] and data_new[i+2]>=data_new[i+4]:
            change_point_max.append(i)
        if data_new[i+2]<=data_new[i] and data_new[i+2]<=data_new[i+1] and data_new[i+2]<=data_new[i+3] and data_new[i+2]<=data_new[i+4]:
            change_point_min.append(i)
    return change_point_max,change_point_min




def conmute_change_point(solution_str, ground_truth):
    ground_truth = extract_ground_truth_values(ground_truth)
    answer = extract_qwen_results(solution_str)

    ground_change_point_max,ground_change_point_min=change_point(ground_truth)
    answer_change_point_max,answer_change_point_min=change_point(answer)


    answer_max_shot=0
    answer_min_shot=0

    for i in range(len(answer_change_point_max)):
        if answer_change_point_max[i] in ground_change_point_max:
            answer_max_shot+=1
        else:
            answer_max_shot+=0
    for i in range(len(answer_change_point_min)):
        if answer_change_point_min[i] in ground_change_point_min:
            answer_min_shot+=1
        else:
            answer_min_shot+=0

    return answer_max_shot/len(ground_change_point_max)*0.1+answer_min_shot/len(ground_change_point_min)*0.1
