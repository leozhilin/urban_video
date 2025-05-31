REASONER_PROMPT = """
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions:

The template for the answer is:
Option: []; Reason: []
where the Option only outputs one option from 'A' to 'E' here, do not output redundant content. Reason explains why you choose this option.

{question}
"""

EVALUATOR_PROMPT = """
Here's a question: {question}
Reason step by step. Evaluate any given answer to this question, be smart, logical, and very critical. Just provide concise feedback.
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
    "You will be asked to creatively and critically improve solutions or answers to problems."
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
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

Improve the variable (concise and accurate answer to the question) using the feedback provided in <FEEDBACK> tags.
Your improved answer must follow this exact format:
Option: []; Reason: []
where the Option only outputs one option from 'A' to 'E', and Reason explains why you choose this option.

Send the improved variable in the following format:

<IMPROVED_VARIABLE>
Option: []; Reason: []
</IMPROVED_VARIABLE>

Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else. Make sure to maintain the exact format specified above.""" 