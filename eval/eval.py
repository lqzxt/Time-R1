import json
import os
import fire
import numpy as np

def main(model_name="qwen7b", desc="eval", dataset_name="ETTh1"):
    input_dir = f"output/{dataset_name}/{model_name}_{desc}/"
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    
    mse_list, mae_list = [], []
    for item in json_files:
        with open(os.path.join(input_dir, item)) as f:
            data_json = [json.loads(line) for line in f]
            
        valid_mse = [result['mse'] for result in data_json if result['mse'] != np.inf]
        valid_mae = [result['mae'] for result in data_json if result['mae'] != np.inf]
        
        if not valid_mse:
            print(f"Dataset: {dataset_name} {item} has no valid results.")
            continue
            
        mse_list.append(np.mean(valid_mse))
        mae_list.append(np.mean(valid_mae))
        
    print(f"Dataset: {dataset_name} Average MSE: {np.mean(mse_list)}")
    print(f"Dataset: {dataset_name} Average MAE: {np.mean(mae_list)}")

def run_all():
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", 
               "exchange", "aqwan", "aqshunyi", "wind", "nasdaq"]
    for dataset in datasets:
        main(dataset_name=dataset)

if __name__ == '__main__':
    fire.Fire(run_all)
