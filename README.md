## 数据集下载指令
git clone git@hf.co:datasets/EmbodiedCity/UrbanVideo-Bench


## 工作目录
export PYTHONPATH=$(pwd):$PYTHON
```shell
.
├── baseline #按UrbanVideo给的提示词实现的代码，理论上会和论文给出的baseline达成一样的结果
│   ├── gemini_baseline.py
│   ├── generate_test_samples.py
│   ├── qwen_baseline.py # 论文提供的代码，模型换成了qwen
│   ├── qwen_result
    ├── video_r1_baseline.py 
│   ├── video_r1_result
│   └── test_samples.jsonl # 完整跑数据集太烧钱耗时了，随机取了100个样本便于日常测试
├── gemini_analysis_results.json
├── print_result.py
├── qwen_test.py
├── README.md
├── results.json
├── src #目前正在尝试，LLM充当planner agent将任务拆解
│   ├── agent_loop.py
│   ├── agent_prompt.py
│   ├── clients
│   ├── function_call.py
│   ├── print_results.py
│   ├── __pycache__
│   └── tools
├── tmp.py
├── UrbanVideo-Bench # 数据集
│   ├── evaluations.jpg
│   ├── MCQ.parquet
│   ├── README.md
│   └── videos
```

## baseline
qwen-vl-max baseline 复现结果（100个样例下）：和论文提供的结果大体一致

| Category | Accuracy | Num |
|----------|----------|-----|
| Action Generation | 0.285714286 | 21 |
| Association Reasoning | 0.666666667 | 3 |
| Causal | 0.4 | 5 |
| Cognitive Map | 0.777777778 | 9 |
| Counterfactual | 0.5 | 2 |
| Duration | 1 | 2 |
| Goal Detection | 0.6 | 5 |
| High-level Planning | 0.285714286 | 7 |
| Landmark Position | 0.538461538 | 13 |
| Object Recall | 0.75 | 4 |
| Progress Evaluation | 0.5 | 12 |
| Proximity | 1 | 2 |
| Scene Recall | 0.5 | 6 |
| Sequence Recall | 1 | 2 |
| Start/End Position | 0 | 2 |
| Trajectory Captioning | 0.4 | 5 |
| **Total** | **0.5** | **100** |

video_r1_7b baseline 复现结果（50个样例下）：7B的效果感觉已经接近qwen-vl-max了

| Category             | Accuracy      | Num |
|----------------------|--------------|-----|
| Action Generation    | 0.428571429  | 14  |
| Association Reasoning| 0            | 1   |
| Causal               | 0.5          | 2   |
| Cognitive Map        | 1            | 2   |
| Duration             | 1            | 1   |
| Goal Detection       | 0.5          | 2   |
| Landmark Position    | 0.142857143  | 7   |
| Object Recall        | 0.5          | 2   |
| Progress Evaluation  | 0.416666667  | 12  |
| Proximity            | 1            | 1   |
| Scene Recall         | 1            | 3   |
| Trajectory Captioning| 0.333333333  | 3   |
| **Total**            | **0.46**     | **50** |