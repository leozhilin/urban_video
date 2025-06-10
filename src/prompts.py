REASONER_PROMPT = """
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions by following these steps:

The question is:
{question}

Problem Description:
The position at the beginning of the video is your initial position. You will receive a series of navigation instructions to move, and the entire video shows the scene of you following the instructions to move.
If the question explicitly provides specific navigation instructions, then your current position is the position after completing this series of instructions. In this case, you need to combine the video content to determine your current position and answer the question based on your current position;
If the question does not have explicit navigation instructions, your current position defaults to the last position in the video, which is the position after executing all navigation instructions.

You need to combine the question and video content to answer the question.
Questions can generally be divided into temporal questions and image questions:

1. Image questions: Questions usually only involve understanding the scene at a single position (for example: at the current position, Is xx on your left, center, or right?).
- Analyze the current position and locate the correct scene/frame corresponding to the question
- Combine the question content and scene content to answer the question
2. Temporal questions: You need to combine the execution of a series of temporal actions to answer the question

You need to think and reason step by step, answer the question according to the following template, and give your reasons:

The template for the answer is:
Option: []; Reason: []
where:
- Option: Choose only one option from 'A' to 'E'
- Reason: Explain your choice by referencing specific visual evidence from the video

"""

EVALUATOR_PROMPT = """
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please note that due to technical limitations, only a subset of video frames is provided, which means some details might be missing from the visual evidence.
Here's a question about this video: {question}

Please evaluate the following answer by:
1. Carefully analyzing the video content and visual evidence
2. Checking if the answer aligns with what is shown in the video frames
3. Identifying any discrepancies between the answer and the video content
4. Providing specific feedback based on the visual evidence

Be smart, logical, and critical in your evaluation. Focus on how well the answer reflects the actual content shown in the video.

Answer to evaluate:
{answer}
"""

FEEDBACKER_PROMPT = """
You are a feedback provider who needs to give clear feedback on the answer based on the language model's evaluation.

Question: {question}

Original Answer: {answer}

Language Model's Evaluation: {evaluation}

Based on the above evaluation, provide structured feedback including:

1. Strengths of the Answer:
   - List the aspects that were done well according to the evaluation
   - Specifically explain how these strengths contribute to answering the question
   - Skip this section if no strengths are explicitly mentioned in the evaluation

2. Areas for Improvement:
   - List specific issues or shortcomings identified in the evaluation
   - Provide concrete improvement suggestions for each issue
   - Ensure suggestions align with the video content

3. Overall Recommendations:
   - Summarize how to improve the answer
   - Emphasize strengths that should be maintained
   - Provide specific directions for improvement

Please ensure:
- Feedback is clear, specific, and well-structured
- Content is strictly based on the evaluation, without adding unmentioned points
- Use concise and clear language
- Maintain an objective and professional tone
"""


GLOSSARY_TEXT = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""

### Optimize Prompts

# System prompt to TGD
OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve solutions or answers to problems. "
    "You will receive a video (captured into multiple frames of images as follows), which presents the perception data of an agent moving in the environment from a first person perspective. "
    "Please note that due to technical limitations, only a subset of video frames is provided, which means some details might be missing from the visual evidence. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy or contain inaccuracies - carefully analyze and critically evaluate each point of feedback. "
    "Always verify feedback against the available video content and only incorporate valid suggestions that align with the visual evidence. "
    "When improving the answer, focus on making it more reasonable and well-supported by the available evidence, even if some details cannot be fully verified. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
    "This is very important: You MUST give your response by sending the improved variable between <IMPROVED_VARIABLE> {{improved variable}} </IMPROVED_VARIABLE> tags. "
    "The text you send between the tags will directly replace the variable.\n\n"
    f"{GLOSSARY_TEXT}"
)

OPTIMIZER_PROMPT = """
Here is the role of the variable you will improve: <ROLE>concise and accurate answer to the question</ROLE>.

The variable is the text within the following span: <VARIABLE> {answer} </VARIABLE>

Here is the context and feedback we got for the variable:

<FEEDBACK> {feedback} </FEEDBACK>

Please improve the answer by following these steps:
1. Carefully review the feedback and identify key areas for improvement
2. Ensure the improved answer strictly aligns with the video content shown in the frames
3. Make the reasoning more concrete by referencing specific visual evidence from the video
4. Maintain the required format while making the answer more accurate and precise

Your improved answer must follow this exact format:
Option: []; Reason: []
where:
- Option: Choose only one option from 'A' to 'E'
- Reason: Explain your choice by referencing specific visual evidence from the video

Send the improved variable in the following format:

<IMPROVED_VARIABLE>
Option: []; Reason: []
</IMPROVED_VARIABLE>

Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else. Make sure to maintain the exact format specified above.
""" 