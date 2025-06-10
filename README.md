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

{
  "video_id": "RealWorld_41.mp4",
  "question": "```\n\u201c\u201c\u201c\nQuestion: You are moving following a series of movement instructions: \"Rotate and observe the surrounding environment. Then move towards the direction of the road between the gray and light blue buildings. Fly along the road for a distance.\" What is the positional relationship between you and the intersection beside the tall building when it reached current position?\nChoices:\nA.\tI am located to the right of the intersection beside the tall building.\nB.\tI am located above the intersection beside the tall building.\nC.\tI am located to the left of the intersection beside the tall building.\nD.\tI am located behind the intersection beside the tall building.\nE.\tI am located in front of the intersection beside the tall building.\n\n\u201d\u201d\u201d\n```\n",
  "question_category": "Landmark Position",
  "ground_truth": "B",
  "textgrad_result": {
    "init_answer": "### Selected Frames: [25, 30, 35, 40, 45]\n\n### Thinking:\n\n1. **Identifying Relevant Frames:**\n   - Frames 25 to 45 show a clear view of the road between the gray and light blue buildings, which is the path the agent is instructed to follow.\n   - These frames capture the movement along the road and the surrounding environment.\n\n2. **Temporal Sequence Analysis:**\n   - The sequence shows a continuous forward movement along the road.\n   - The agent's position relative to the intersection beside the tall building can be observed as it progresses.\n\n3. **Spatial and Temporal Changes:**\n   - **Movement Patterns and Directions:** The agent moves straight along the road, maintaining a consistent direction.\n   - **Position Changes:** The agent approaches and passes the intersection beside the tall building.\n   - **Key Visual Elements:** The tall building is on the right side of the road, and the intersection is clearly visible.\n   - **State Changes or Interactions:** No significant interactions occur; the focus is on the movement and positional relationship.\n\n### Visual Evidence-based Reasoning:\n\n- **Examining the Identified Video Segments:**\n  - In Frame 25, the agent is approaching the intersection.\n  - By Frame 30, the agent is at the intersection.\n  - In Frame 35, the agent has passed the intersection and is moving further along the road.\n  - Frames 40 and 45 continue to show the agent moving away from the intersection.\n\n- **Comparing Each Option Against the Actual Video Content:**\n  - **Option A:** \"I am located to the right of the intersection beside the tall building.\" - This is incorrect as the agent is not to the right but passing through the intersection.\n  - **Option B:** \"I am located above the intersection beside the tall building.\" - This is incorrect as the agent is at ground level.\n  - **Option C:** \"I am located to the left of the intersection beside the tall building.\" - This is incorrect as the agent is not to the left but passing through.\n  - **Option D:** \"I am located behind the intersection beside the tall building.\" - This is correct as the agent has passed the intersection.\n  - **Option E:** \"I am located in front of the intersection beside the tall building.\" - This is incorrect as the agent has already passed the intersection.\n\n### Option: D\n\n### Reason:\n\n- **Specific Visual Evidence:**\n  - In Frame 35, the agent has clearly passed the intersection, indicating it is now behind the intersection.\n  - The tall building remains on the right side, confirming the agent's position relative to the intersection.\n- **Temporal Sequence Support:**\n  - The continuous forward movement from Frame 25 to Frame 45 supports the conclusion that the agent is now behind the intersection.\n\nThus, the most accurate answer based on the visual evidence is **Option D: I am located behind the intersection beside the tall building.**",
    "evaluation": "### Evaluation of the Answer\n\n#### 1. Carefully Analyzing the Video Content and Visual Evidence\n\nThe provided video frames depict an aerial view of an urban environment with various buildings, roads, and green spaces. The sequence shows a continuous forward movement along a road, with the agent (camera) passing by different structures and intersections.\n\n#### 2. Checking if the Answer Aligns with What is Shown in the Video Frames\n\nThe answer provided focuses on the positional relationship between the agent and an intersection beside a tall building. However, the video frames do not explicitly show a \"tall building\" or a specific \"intersection\" as described in the question. Instead, the frames show a general urban landscape with multiple buildings and roads.\n\n#### 3. Identifying Any Discrepancies Between the Answer and the Video Content\n\n- **Tall Building:** The video does not prominently feature a \"tall building\" that stands out from the rest. The buildings appear to be of similar height, with none distinctly taller than the others.\n- **Intersection:** While there are intersections visible in the video, there is no clear indication of an \"intersection beside the tall building\" as mentioned in the question.\n- **Positional Relationship:** The answer concludes that the agent is \"behind the intersection beside the tall building.\" However, without a clear reference to a specific intersection and tall building, this conclusion is not supported by the visual evidence.\n\n#### 4. Providing Specific Feedback Based on the Visual Evidence\n\n- **Relevance of Frames:** The selected frames (25, 30, 35, 40, 45) show a continuous forward movement along a road, but they do not provide enough context to determine the exact positional relationship with a specific intersection and tall building.\n- **Accuracy of Reasoning:** The reasoning in the answer assumes the existence of a \"tall building\" and a specific \"intersection,\" which are not clearly identifiable in the provided frames. Therefore, the conclusion drawn may not be accurate based on the given visual evidence.\n\n### Conclusion\n\nThe answer provided does not align well with the visual evidence from the video frames. The frames do not clearly show a \"tall building\" or a specific \"intersection\" as described in the question. Therefore, the conclusion that the agent is \"behind the intersection beside the tall building\" is not supported by the provided visual evidence.\n\n### Suggested Improvement\n\nTo improve the answer, it would be necessary to:\n\n1. **Re-evaluate the Question:** Ensure that the question accurately reflects the content of the video.\n2. **Analyze the Frames More Closely:** Identify specific features in the frames that can be used to determine the positional relationship.\n3. **Provide Clear Evidence:** Base the conclusion on clear and identifiable features in the video frames.\n\nGiven the current frames and the question, a more accurate answer might focus on the general movement along the road and the relative positions of the buildings and intersections visible in the frames. \n\n**Final Answer:**\n\nBased on the provided video frames, the positional relationship between the agent and any specific intersection or tall building cannot be definitively determined. The frames show a continuous forward movement along a road with multiple buildings and intersections, but no distinct \"tall building\" or specific \"intersection\" is clearly identifiable. Therefore, the most accurate conclusion is that the agent is moving along a road between various buildings and intersections, but the exact positional relationship as described in the question cannot be confirmed.",
    "feedback": "### Feedback on the Answer\n\n#### 1. Strengths of the Answer:\n\n- **Detailed Analysis of Movement:** The answer provides a thorough analysis of the agent's movement along the road, which is a strength. It correctly identifies the continuous forward motion and the general path taken by the agent.\n- **Temporal Sequence Understanding:** The temporal sequence of the frames is well understood, with clear indications of the agent's position at different points in time (Frames 25, 30, 35, 40, 45).\n\n#### 2. Areas for Improvement:\n\n- **Lack of Specific Visual Evidence:** The evaluation notes that the video frames do not explicitly show a \"tall building\" or a specific \"intersection\" as described in the question. This discrepancy needs to be addressed.\n  - **Improvement Suggestion:** Re-examine the frames for any features that could be interpreted as a \"tall building\" or \"intersection.\" If such features are not clearly identifiable, acknowledge this in the answer.\n- **Assumptions Without Clear Support:** The conclusion that the agent is \"behind the intersection beside the tall building\" is based on assumptions not fully supported by the visual evidence.\n  - **Improvement Suggestion:** Avoid making assumptions not directly supported by the video content. Instead, focus on what can be definitively observed in the frames.\n- **Alignment with Question Requirements:** The answer does not fully align with the question due to the lack of specific visual evidence for the \"tall building\" and \"intersection.\"\n  - **Improvement Suggestion:** Ensure that the answer directly addresses the question based on the available visual evidence. If necessary, state that the specific positional relationship cannot be determined from the given frames.\n\n#### 3. Overall Recommendations:\n\n- **Focus on Observable Features:** Concentrate on the features that are clearly observable in the video frames. Describe the agent's movement relative to these observable features.\n- **Acknowledge Limitations:** If the video frames do not provide enough information to determine the exact positional relationship, explicitly state this limitation.\n- **Maintain Strengths:** Keep the detailed analysis of movement and temporal sequence, as these aspects contribute positively to understanding the agent's path.\n- **Specific Directions for Improvement:**\n  - Re-evaluate the frames for any potential \"tall building\" or \"intersection.\"\n  - Adjust the conclusion to reflect only what can be definitively observed.\n  - Clearly communicate any limitations in determining the exact positional relationship.\n\n### Revised Answer Example:\n\nBased on the provided video frames, the agent is observed moving along a road between various buildings. The frames show a continuous forward motion, with the agent passing through intersections and alongside different structures. However, the specific \"tall building\" and \"intersection\" mentioned in the question are not clearly identifiable in the frames.\n\n- **Movement Analysis:** The agent moves from Frame 25 to Frame 45, consistently following the road. By Frame 35, the agent has passed an intersection, indicating it is now behind this intersection.\n- **Positional Relationship:** While the exact positional relationship with a \"tall building\" cannot be determined, the agent is clearly behind an intersection by Frame 35.\n\nGiven the limitations in identifying the \"tall building,\" the most accurate conclusion is that the agent is **behind an intersection** by the current position, but the specific relationship with the \"tall building\" remains unclear.\n\n**Final Answer: D (with noted limitations).**\n\nThis revised approach maintains the strengths of the original answer while addressing the identified areas for improvement.",
    "final_answer": "Option: D; Reason: Based on the provided video frames, the agent is observed moving along a road between various buildings. The frames show a continuous forward motion, with the agent passing through intersections and alongside different structures. However, the specific \"tall building\" and \"intersection\" mentioned in the question are not clearly identifiable in the frames. By Frame 35, the agent has passed an intersection, indicating it is now behind this intersection. Therefore, the most accurate conclusion is that the agent is **behind an intersection** by the current position, but the specific relationship with the \"tall building\" remains unclear."
  }
}