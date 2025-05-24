# 数据集下载指令
git clone git@hf.co:datasets/EmbodiedCity/UrbanVideo-Bench

# 工作目录
export PYTHONPATH=$(pwd):$PYTHON

.
├── baseline #按UrbanVideo给的提示词实现的代码，理论上会和论文给出的baseline达成一样的结果
│   ├── gemini_baseline.py
│   ├── generate_test_samples.py
│   ├── qwen_baseline.py # 论文提供的代码，模型换成了qwen
│   ├── qwen_result
│   └── test_samples.jsonl # 完整跑数据集太烧钱耗时了，随机取了100个样本便于日常测试
├── gemini_analysis_results.json
├── print_result.py
├── qwen_test.py
├── README.md
├── results.json
├── src #目前正在尝试的手动，LLM充当planner agent将任务拆解
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