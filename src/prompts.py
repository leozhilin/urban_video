REASONER_PROMPT = """
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions by following these steps:

1. Video Segment Analysis:
   - First, carefully analyze the question to determine:
     * What specific moment or period in the video is being asked about (e.g., initial position, a specific action, etc.)
     * What visual elements or conditions need to be verified
   - Then, identify ONLY the relevant frame range:
     * Focus on the specific continuous sequence of frames that directly relate to the question
     * Do not include frames before or after the relevant period
     * For example, if the question asks about the initial position, only analyze the first few frames
   - For the identified frames, analyze:
     * The spatial and temporal context (e.g., position, orientation, movement)
     * Key visual elements and their states
     * Any relevant changes or conditions within this specific frame range

2. Visual Evidence-based Reasoning:
   - Carefully examine the identified video segments
   - Compare each option against the actual video content
   - Select the option that is most strongly supported by visual evidence
   - Provide specific visual evidence from the frames to justify your choice

The question is:
{question}

The template for the answer is:
Selected Frames: [List the frame numbers that are most relevant to the question];
Thinking: [Describe your analysis process, including:
  * How you identified these specific frames
  * What key observations you made in these frames
  * How these observations relate to the question];
Option: []; Reason: []
where:
- Option: Choose only one option from 'A' to 'E'
- Reason: Explain your choice by referencing specific visual evidence from the video, including:
  * Which frames support your choice
  * What specific visual elements or changes in these frames support your reasoning
  * How the temporal sequence of events relates to your choice

"""

EVALUATOR_PROMPT = """
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective.
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
You will give feedback to a variable with the following role: 
<ROLE>concise and accurate answer to the question</ROLE>

Here is an evaluation of the variable using a language model:

<LM_SYSTEM_PROMPT> Here's a question: {question} Evaluate any given answer to this question, be smart, logical, and very critical. Just provide concise feedback. </LM_SYSTEM_PROMPT>

<LM_INPUT> {answer} </LM_INPUT>

<LM_OUTPUT> {evaluation} </LM_OUTPUT>

<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>

We are interested in giving feedback to the concise and accurate answer to the question for this conversation. Specifically, give feedback to the following span of text:

<VARIABLE> {answer} </VARIABLE>

Given the above history, describe how the concise and accurate answer to the question could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.
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
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy or contain inaccuracies - carefully analyze and critically evaluate each point of feedback. "
    "Always verify feedback against the video content and only incorporate valid suggestions that align with the visual evidence. "
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