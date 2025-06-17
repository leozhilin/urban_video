few_shot_data = [
    {
        "video_id": "AerialVLN_10_3HYA4D4520HIRF1OVH7WHN46KPYF2W_5.mp4",
        "question": "Question: Navigation Instruction given at initial position: [take off to the ground near the flag and turn left move upward to reach the terrace of the building. to reach the antenna in that building and cross the building to the next building to see the red colour eds butcher shop board in it. to move forward to see the pool shines by sunlight and cross the pool to reach the street. move towards to the top of the building and see the cloudy, shiny sky and also the reflected sea. turn left side and move towards the pool and fully surrounded by red leaves of trees and move towards to the street. move to the same road on the street and go straight to the corner of the road and reach the optician board on the right side. and move down to the board and see the health insurance in the down to the optician in that board.]. At the initial position, you are asked to navigate according to a series of movement instructions. When you reach the final position, what is your orientation to the pool?  \nchoose:  \nA. I am on the left side of the pool.  \nB. I am located below the pool.  \nC. I am on the right side of the pool.  \nD. I am located above the pool.  \nE. I am on the pool floor.",
        "question_category": "Landmark Position",
        "textgrad_result": {
            "init_answer": "Option: D; Reason: The question asks about my orientation relative to the pool at the final position. Therefore, I need to determine my relative position to the pool from the last few frames of the video. In the final frames, the pool is located below in the frame, and I am moving upward relative to the pool. Therefore, it can be determined that I am located above the pool.",
            "evaluation": "Location: [(243, 255)]\nOption: A\nReason: Since the question asks about the final position, we need to look at the end of the video for the answer. In the final 243~255s of the video, I am moving upward, with a pool surrounded by red trees below in the frame, and city buildings ahead. Since the question asks about my relative position to the pool at this moment, it can be determined that I am located above the pool, which matches the description in option D. The Init Answer's analysis shows that it also located the answer at the end of the video and chose the same option as my analysis, with thorough reasoning and no logical errors.",
        }
    },
    {
        "video_id": "RealWorld_26.mp4",
        "question": "Question: At the initial position, you are asked to navigate to the outer edge of the community and circle half a circle. At your current location, if the outer edge of the residential area is not within your field of view, what is the most relevant object in your field of view and where is this object?  \nchoose:  \nA. I can see the outer edge of the residential area.  \nB. I cannot see the outer edge of the residential area, but there is a busy main road in the lower right corner of my field of view.  \nC. I cannot see the outer edge of the residential area, but there is a construction site in the lower right corner of my field of view.  \nD. I cannot see the outer edge of the residential area, but there is a lake at the bottom of my field of view.  \nE. I cannot see the outer edge of the residential area, but the parking lot is in the lower right corner of my field of view.  ",
        "question_category": "Association Reasoning",
        "ground_truth": "A",
        "textgrad_result": {
            "init_answer": "Option: C; Reason: The video frames show a construction site in the lower right corner of the field of view, which is consistent with the description in option C. The construction site is visible with structures and materials that are typical of such an area. The other options do not match the visual content of the frames provided.",
            "evaluation": "Location: [(0, 13), (27, 31), (35, 47), (127, 135), (135, 147), (151, 160), (199, 222)]\nOption: A\nReason: At the initial position, multiple tall buildings can be seen in the field of view, with trees and paths between them, which suggests that this is a residential area. Moving forward, you can see city streets and a pond. From a bird’s-eye view, you can see that the streets and pond are separated from the residential area by a fence, which should be the edge of the residential area. As the camera moves to the right, a low building can be seen on the left, and further ahead there is a construction site, which is also separated from the residential area by a fence. Continuing to fly forward, the residential area always remains on the right side of the frame, and a fence consistently appears in the center of the view, indicating that the navigation instruction to \"navigate to the outer edge of the community and circle half a circle\" is being executed. Although the construction site does appear in the frame, saying \"cannot see the outer edge of the residential area\" is incorrect, because a fence clearly marking the edge of the residential area is visible in the video. Based on this, Option A is the most correct answer.",
        }
    }
]   

from prompts import EVALUATOR_PROMPT
def make_few_shot_msg(few_shot_data: list[dict], question_category: str):
    target_data = [item for item in few_shot_data if item['question_category'] == question_category]

    example_content = []
    for example in target_data[:1]:
        evaluator_prompt = EVALUATOR_PROMPT.format(
            question=example["question"],
            answer=example["textgrad_result"]["init_answer"],
            question_category=example["question_category"]
        )
        evaluation = example["textgrad_result"]["evaluation"]
        example_content += [
            {"type": "text", "text": "The following is a reference example. The video has been omitted for cost considerations, but you can refer to its reasoning and analysis process."},
            {"type": "text", "text": evaluator_prompt},
            {"type": "text", "text": evaluation},
        ]

    return example_content

if __name__ == "__main__":
    print(make_few_shot_msg(few_shot_data, "Association Reasoning"))