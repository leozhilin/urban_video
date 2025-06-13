REASONER_PROMPT = """
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions:
The template for the answer is: 
   Option: []; Reason: []
   where the Option only outputs one option from 'A' to 'E' here, do not output redundant content. Reason explains why you choose this option.
{question}

"""

EVALUATOR_PROMPT = """ # 改用gemini， 也要求以固定格式输出答案，如果两者答案一致跳过后续步骤
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. 
Here's a question about this video: {question}

Please evaluate the following answer by: 
1. Based on your analysis of the video content, locate the video frames that may contain the answer to the question (for example, if the question asks "What objects did you observe when descending towards the community entrance?", you need to locate the frames showing the descent towards the community entrance and focus your analysis on these frames)
2. Carefully analyzing the video content and visual evidence based on the video frames you have located
3. Checking if the answer aligns with what is shown in the video frames
4. Identifying any discrepancies between the answer and the video content
5. Providing specific feedback based on the visual evidence

Be smart, logical, and critical in your evaluation. Focus on how well the answer reflects the actual content shown in the video.

Answer to evaluate:
{answer}
"""

FEEDBACKER_PROMPT = """
You are a viewpoint recorder who needs to document key disagreements and areas requiring further verification between the answer and the evaluation.

Question: {question}

Original Answer: {answer}

Language Model's Evaluation: {evaluation}

Based on the above answer and evaluation, provide a structured analysis including:

1. Key Disagreements:
   - List specific points where the answer and evaluation differ
   - For each disagreement, clearly state both viewpoints
   - Highlight which aspects are supported by visual evidence and which are not

2. Required Verifications:
   - Identify specific visual evidence that needs to be re-examined
   - List additional frames or perspectives that would help resolve disagreements
   - Specify what kind of visual information would be needed to verify each point

3. Resolution Path:
   - Suggest specific steps to resolve each disagreement
   - Prioritize which verifications are most critical
   - Outline what additional evidence would be most helpful

Please ensure:
- Focus on factual disagreements rather than subjective opinions
- Be specific about what visual evidence is needed
- Maintain a neutral and objective tone
- Structure the analysis clearly and logically
"""


# System prompt to TGD
OPTIMIZER_SYSTEM_PROMPT = (
   ""
)

OPTIMIZER_PROMPT = """
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. 

Here's a question about this video: {question}

Original Answer: {answer}

Analysis of Disagreements and Required Verifications:
{feedback}

Based on the above analysis, please:
1. Carefully review each identified disagreement
2. Re-examine the video frames, paying special attention to the specific visual evidence mentioned in the analysis
3. Consider the suggested verification steps and additional evidence needed
4. Provide a revised answer that addresses the identified issues

Your revised answer must follow this exact format:
Option: []; Reason: []
where:
- Option: Choose only one option from 'A' to 'E'
- Reason: Explain your choice by:
  * Addressing the key disagreements identified
  * Referencing specific visual evidence that supports your choice
  * Acknowledging any remaining uncertainties
  * Explaining why your choice is the most reasonable given the available evidence
""" 