import json
import os
import threading
import fire
import numpy as np
from tqdm import tqdm

from dataLoader import DataLoader
from utils import get_result, call_R1


def handle_data(model_name, data_chunk, output_dir, run_num, pred_window, progress, lock):
    for item in data_chunk:
        prompt = item["prompt"]
        test_input = item["test_input"]
        ground_truth = item["ground_truth"]
        attr = item["attr"]
        index = item["index"]

        for i in range(run_num):
            try:
                qwen_output = call_R1(prompt)
                qwen_res = get_result(qwen_output)
                if len(qwen_res) == 0:
                    qwen_output += '\n```\n'
                    qwen_res = get_result(qwen_output)
                if len(qwen_res) >= len(ground_truth):
                    qwen_res = qwen_res[0:pred_window]

                    MSE = np.mean((np.array(ground_truth) - np.array(qwen_res)) ** 2)
                    MAE = np.mean(np.abs(np.array(ground_truth) - np.array(qwen_res)))
                else:
                    MSE = np.inf
                    MAE = np.inf
                out_to_file_data = {
                    "mse": MSE,
                    "mae": MAE,
                    "length": len(qwen_res),
                    "prompt": prompt,
                    "qwen_output": qwen_output,
                }
                with open(f"{output_dir}/{attr}_{index}.jsonl", "a") as f:
                    f.write(json.dumps(out_to_file_data) + "\n")
            except Exception as e:
                print(f"Error occurred: {e}")
            finally:
                with lock:
                    progress.update(1)


def main(model_name="deepseek-R1", desc="Base_RL_96_96_n16_oldRL_ETTh1", dataset_name="ETTh1", look_back=96, pred_window=96,
         run_num=1, noTime=False):
    dataLoader = DataLoader(dataset_name, look_back, pred_window, noTime)
    data = dataLoader.load_data()
    output_dir = f"output/{dataset_name}/{model_name}_{desc}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_tasks = len(data) * run_num
    progress = tqdm(total=total_tasks, desc="Processing")
    lock = threading.Lock()

    data_length = len(data)
    thread_candidates = [60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]
    selected_threads = 10
    for candidate in thread_candidates:
        if candidate <= data_length and data_length % candidate == 0:
            selected_threads = candidate
            break
    print(f"Number of threads used: {selected_threads}")

    threads = []
    chunk_size = len(data) // selected_threads

    for i in range(selected_threads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        data_chunk = data[start_idx:end_idx]
        t = threading.Thread(
            target=handle_data,
            args=(model_name, data_chunk, output_dir, run_num, pred_window, progress, lock),
            name=f"Thread-{i}"
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    progress.close()
    print("All threads have completed processing.")


def run_all():
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange", "aqwan", "aqshunyi", "wind", "nasdaq"]
    for dataset in datasets:
        if dataset == "nasdaq":
            main(dataset_name=dataset, look_back=36, pred_window=36)
        else:
            main(dataset_name=dataset)


if __name__ == '__main__':
    fire.Fire(run_all)
