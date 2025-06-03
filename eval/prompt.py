prompt_template = """Here is the {attr_meaning} data of the {dataset_name} dataset. I will now give you data for the past {look_back} recorded dates, and please help me forecast the data for next {pred_window} recorded dates.The data is as follows:
```
{data_lookback}
```
Please give me the complete data for the next {pred_window} recorded dates, remember to give me the complete data.
You must first conduct reasoning inside <think> ...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.
Output format for your answer is :
<think>
...
</think>
<answer>
```
...
```
</answer>
Please obey the format strictly."""
