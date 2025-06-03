import pandas as pd

from prompt import prompt_template

data_dict = {
    'ETTh1': {
        'path': 'dataset/ETT/ETTh1.csv',
        'meaning': {
            'HUFL': 'High UseFul Load', 'HULL': 'High UseLess Load', 'MUFL': 'Middle UseFul Load',
            'MULL': 'Middle UseLess Load', 'LUFL': 'Low UseFul Load', 'LULL': 'Low UseLess Load',
            'OT': 'Oil Temperature'
        }
    },
    'ETTh2': {
        'path': 'dataset/ETT/ETTh2.csv',
        'meaning': {
            'HUFL': 'High UseFul Load', 'HULL': 'High UseLess Load', 'MUFL': 'Middle UseFul Load',
            'MULL': 'Middle UseLess Load', 'LUFL': 'Low UseFul Load', 'LULL': 'Low UseLess Load',
            'OT': 'Oil Temperature'
        }
    },
    'ETTm1': {
        'path': 'dataset/ETT/ETTm1.csv',
        'meaning': {
            'HUFL': 'High UseFul Load', 'HULL': 'High UseLess Load', 'MUFL': 'Middle UseFul Load',
            'MULL': 'Middle UseLess Load', 'LUFL': 'Low UseFul Load', 'LULL': 'Low UseLess Load',
            'OT': 'Oil Temperature'
        }
    },
    'ETTm2': {
        'path': 'dataset/ETT/ETTm2.csv',
        'meaning': {
            'HUFL': 'High UseFul Load', 'HULL': 'High UseLess Load', 'MUFL': 'Middle UseFul Load',
            'MULL': 'Middle UseLess Load', 'LUFL': 'Low UseFul Load', 'LULL': 'Low UseLess Load',
            'OT': 'Oil Temperature'
        }
    },
    'exchange': {
        'path': 'dataset/exchange/exchange.csv',
        'meaning': {
            'Australia': 'the daily exchange rates of Australia',
            'British': 'the daily exchange rates of British',
            'Canada': 'the daily exchange rates of Canada',
            'Switzerland': 'the daily exchange rates of Switzerland',
            'China': 'the daily exchange rates of China',
            'Japan': 'the daily exchange rates of Japan',
            'New_Zealand': 'the daily exchange rates of New Zealand',
            'Singapore': 'the daily exchange rates of Singapore'
        }
    },
    'aqwan': {
        'path': 'dataset/aqwan/aqwan.csv',
        'meaning': {
            'CO': 'Carbon Monoxide',
            'DEWP': 'Dew Point',
            'NO2': 'Nitrogen Dioxide',
            'O3': 'Ozone',
            'PM10': 'Particulate Matter 10',
            'PM2.5': 'Particulate Matter 2.5',
            'PRES': 'Pressure',
            'RAIN': 'Rainfall',
            'SO2': 'Sulfur Dioxide',
            'TEMP': 'Temperature',
            'WSPM': 'Wind Speed'
        }
    },
    'aqshunyi': {
        'path': 'dataset/aqshunyi/aqshunyi.csv',
        'meaning': {
            'CO': 'Carbon Monoxide',
            'DEWP': 'Dew Point',
            'NO2': 'Nitrogen Dioxide',
            'O3': 'Ozone',
            'PM10': 'Particulate Matter 10',
            'PM2.5': 'Particulate Matter 2.5',
            'PRES': 'Pressure',
            'RAIN': 'Rainfall',
            'SO2': 'Sulfur Dioxide',
            'TEMP': 'Temperature',
            'WSPM': 'Wind Speed'
        }
    },
    'wind': {
        'path': 'dataset/wind/wind.csv',
        'meaning': {
            'pred_humidity': 'Predicted Humidity',
            'pred_pressure': 'Predicted Pressure',
            'pred_temp': 'Predicted Temperature',
            'pred_w_dir': 'Predicted Wind Direction',
            'pred_w_speed': 'Predicted Wind Speed',
            'target': 'Target',
            'ture_w_speed': 'True Wind Speed'
        }
    },
    'nasdaq': {
        'path': 'dataset/nasdaq/nasdaq.csv',
        'meaning': {
            'Close': 'closing price',
            'Open': 'opening price',
            'Volume': 'trading volume',
            'Low': 'lowest price',
            'High': 'highest price'
        }
    }
}


class DataLoader:
    def __init__(self, dataset_name, look_back, pred_window, noTime):
        if dataset_name in data_dict:
            self.dataset_name = dataset_name
            self.data_path = data_dict[dataset_name]['path']
            self.meaning = data_dict[dataset_name]['meaning']
            self.look_back = look_back
            self.pred_window = pred_window
            self.prompt_template = prompt_template
            self.noTime = noTime
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")

    def load_data(self):
        res = []
        original_data = pd.read_csv(self.data_path)
        if 'ETT' in self.dataset_name:
            for attr in self.meaning.keys():
                if self.dataset_name == 'ETTh1' or self.dataset_name == 'ETTh2':
                    data = original_data[12 * 30 * 24 + 4 * 30 * 24 - self.look_back: 12 * 30 * 24 + 8 * 30 * 24]
                else:
                    data = original_data[
                           12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.look_back: 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
                date = data.loc[:, 'date'].to_numpy()
                attr_data = data.loc[:, attr].to_numpy()

                if self.noTime:
                    data = pd.DataFrame()
                else:
                    data = pd.DataFrame(date, columns=['date'])
                data[attr] = attr_data

                if self.dataset_name == 'ETTh1' or self.dataset_name == 'ETTh2':
                    border1s = [0, 12 * 30 * 24 - self.look_back, 12 * 30 * 24 + 4 * 30 * 24 - self.look_back]
                    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
                else:
                    border1s = [0, 12 * 30 * 24 * 4 - self.look_back,
                                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.look_back]
                    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
                gt = original_data[border1s[2]:border2s[2]]
                gt = gt[attr].values

                for i in range(10):
                    test_input = data.iloc[i * self.look_back:(i + 1) * self.look_back]
                    if self.noTime:
                        test_input.insert(0, 'index', range(1, len(test_input) + 1))
                    ground_truth = gt[(i + 1) * self.look_back:(i + 1) * self.look_back + self.pred_window]
                    prompt = self.prompt_template.replace("{attr_meaning}", self.meaning[attr]).replace(
                        "{dataset_name}", self.dataset_name).replace("{look_back}", str(self.look_back)).replace(
                        "{pred_window}", str(self.pred_window)).replace("{data_lookback}",
                                                                        test_input.to_string(index=False))
                    res.append({
                        'prompt': prompt,
                        'test_input': test_input,
                        'ground_truth': ground_truth,
                        'attr': attr,
                        'index': i
                    })

        else:
            num_train = int(len(original_data) * 0.7)
            num_test = int(len(original_data) * 0.2)
            data_progress = original_data[len(original_data) - num_test - self.look_back:len(original_data)]
            date = data_progress.loc[:, 'date'].to_numpy()
            for attr in self.meaning.keys():
                attr_data = data_progress.loc[:, attr].to_numpy().round(3)

                if self.noTime:
                    data = pd.DataFrame()
                else:
                    data = pd.DataFrame(date, columns=['date'])

                data[attr] = attr_data

                num_train = int(len(original_data) * 0.7)
                num_test = int(len(original_data) * 0.2)
                data_gt = original_data[len(original_data) - num_test - self.look_back: len(original_data)]
                data_gt = data_gt[attr].values

                for i in range(10):
                    if self.dataset_name == 'nasdaq':
                        test_input = data.iloc[i: i + self.look_back]
                        ground_truth = data_gt[i + self.look_back:i + self.look_back + self.pred_window]
                    else:
                        test_input = data.iloc[i * self.look_back:(i + 1) * self.look_back]
                        ground_truth = data_gt[(i + 1) * self.look_back:(i + 1) * self.look_back + self.pred_window]

                    if self.noTime:
                        test_input.insert(0, 'index', range(1, len(test_input) + 1))

                    prompt = self.prompt_template.replace("{attr_meaning}", self.meaning[attr]).replace(
                        "{dataset_name}", self.dataset_name).replace("{look_back}", str(self.look_back)).replace(
                        "{pred_window}", str(self.pred_window)).replace("{data_lookback}",
                                                                        test_input.to_string(index=False))
                    res.append({
                        'prompt': prompt,
                        'test_input': test_input,
                        'ground_truth': ground_truth,
                        'attr': attr,
                        'index': i
                    })
        return res

    def reGeneratePrompt(self, current_index, current_attr, new_data):
        res = []
        original_data = pd.read_csv(self.data_path)
        if 'ETT' in self.dataset_name:
            for attr in self.meaning.keys():
                if self.dataset_name == 'ETTh1' or self.dataset_name == 'ETTh2':
                    data = original_data[12 * 30 * 24 + 4 * 30 * 24 - self.look_back: 12 * 30 * 24 + 8 * 30 * 24]
                else:
                    data = original_data[
                           12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.look_back: 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
                date = data.loc[:, 'date'].to_numpy()
                attr_data = data.loc[:, attr].to_numpy()

                if self.noTime:
                    data = pd.DataFrame()
                else:
                    data = pd.DataFrame(date, columns=['date'])
                data[attr] = attr_data

                if self.dataset_name == 'ETTh1' or self.dataset_name == 'ETTh2':
                    border1s = [0, 12 * 30 * 24 - self.look_back, 12 * 30 * 24 + 4 * 30 * 24 - self.look_back]
                    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
                else:
                    border1s = [0, 12 * 30 * 24 * 4 - self.look_back,
                                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.look_back]
                    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
                gt = original_data[border1s[2]:border2s[2]]
                gt = gt[attr].values

                for i in range(10):
                    if current_index != i or current_attr != attr:
                        continue

                    test_input = data.iloc[i * self.look_back:(i + 1) * self.look_back]
                    if self.noTime:
                        test_input.insert(0, 'index', range(1, len(test_input) + 1))
                    
                    if new_data is not None:
                        test_input.loc[:, attr] = new_data
                    
                    ground_truth = gt[(i + 1) * self.look_back:(i + 1) * self.look_back + self.pred_window]
                    prompt = self.prompt_template.replace("{attr_meaning}", self.meaning[attr]).replace(
                        "{dataset_name}", self.dataset_name).replace("{look_back}", str(self.look_back)).replace(
                        "{pred_window}", str(self.pred_window)).replace("{data_lookback}",
                                                                        test_input.to_string(index=False))
                    return prompt

        else:
            num_train = int(len(original_data) * 0.7)
            num_test = int(len(original_data) * 0.2)
            data_progress = original_data[len(original_data) - num_test - self.look_back:len(original_data)]
            date = data_progress.loc[:, 'date'].to_numpy()
            for attr in self.meaning.keys():
                attr_data = data_progress.loc[:, attr].to_numpy().round(3)

                if self.noTime:
                    data = pd.DataFrame()
                else:
                    data = pd.DataFrame(date, columns=['date'])

                data[attr] = attr_data

                num_train = int(len(original_data) * 0.7)
                num_test = int(len(original_data) * 0.2)
                data_gt = original_data[len(original_data) - num_test - self.look_back: len(original_data)]
                data_gt = data_gt[attr].values

                for i in range(10):
                    if current_index != i or current_attr != attr:
                        continue
                    if self.dataset_name == 'nasdaq':
                        test_input = data.iloc[i: i + self.look_back]
                        ground_truth = data_gt[i + self.look_back:i + self.look_back + self.pred_window]
                    else:
                        test_input = data.iloc[i * self.look_back:(i + 1) * self.look_back]
                        ground_truth = data_gt[(i + 1) * self.look_back:(i + 1) * self.look_back + self.pred_window]

                    if self.noTime:
                        test_input.insert(0, 'index', range(1, len(test_input) + 1))
                    
                    if new_data is not None:
                        test_input.loc[:, attr] = new_data

                    prompt = self.prompt_template.replace("{attr_meaning}", self.meaning[attr]).replace(
                        "{dataset_name}", self.dataset_name).replace("{look_back}", str(self.look_back)).replace(
                        "{pred_window}", str(self.pred_window)).replace("{data_lookback}",
                                                                        test_input.to_string(index=False))
                    return prompt