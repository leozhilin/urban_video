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


### 提示词优化方法
#### 推理者
```
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions:

The template for the answer is:
Option: []; Reason: []
where the Option only outputs one option from 'A' to 'E' here, do not output redundant content. Reason explains why you choose this option.

[question]
```

#### 评估者
```
System:
Here's a question: [原始问题]
Reason step by step. Evaluate any given answer to this question, be smart, logical, and very critical. Just provide concise feedback.
[Reasoner提供的初始回答]
```
#### 反馈者
```
You will give feedback to a variable with the following role: 
<ROLE> ... </ROLE>.
 Here is an evaluation of the variable using a language model:
 
<LM_SYSTEM_PROMPT> Here's a question: ... Evaluate any given answer to this question, be smart, logical, and very critical. Just provide concise feedback. </LM_SYSTEM_PROMPT>
 
<LM_INPUT> [evaluator的输入] </LM_INPUT>
 
<LM_OUTPUT> [evaluator的输出] </LM_OUTPUT>
 
<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>
 
We are interested in giving feedback to the concise and accurate answer to the question for this conversation. Specifically, give feedback to the following span of text:
 
<VARIABLE> [原始回答内容] </VARIABLE>
 
Given the above history, describe how the concise and accurate answer to the question could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.

```

#### 优化者
```
Here is the role of the variable you will improve: <ROLE>concise and accurate answer to the question</ROLE>.

The variable is the text within the following span: <VARIABLE> [original answer content] </VARIABLE>

Here is the context and feedback we got for the variable:

[Contains previous conversation and feedback]

Improve the variable (concise and accurate answer to the question) using the feedback provided in <FEEDBACK> tags.
Send the improved variable in the following format:

<IMPROVED_VARIABLE>{the improved variable}</IMPROVED_VARIABLE>

Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else.
```