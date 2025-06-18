import os
import cv2
import base64
import json
import pandas as pd
from src.textgrad_urban import UrbanTextGrad
import math

def read_video_frames(video_path: str, max_frames: int = 32) -> list:
    """Read video frames and convert to base64."""
    video = cv2.VideoCapture(video_path)
    frames = []
    
    # 首先读取所有帧
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        _, buffer = cv2.imencode(".jpg", frame)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    video.release()
    
    # 如果帧数超过max_frames，进行均匀采样
    if len(frames) > max_frames:
        div_num = math.ceil(len(frames) / max_frames)  # 计算采样间隔
        frames = frames[0::div_num]  # 均匀采样
    
    return frames

def main():
    # Initialize TextGrad with your model
    # textgrad = UrbanTextGrad(model_name="qwen-vl-max-latest")
    textgrad = UrbanTextGrad()
    
    # Load test samples
    with open('baseline/test_samples.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    QA_df = pd.DataFrame(data)
    
    # Create results directory if it doesn't exist
    results_dir = 'textgrad_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each sample
    for idx, row in QA_df.iterrows(): 
        # if row['question_category'] == "Action Generation":
        if idx in [158, 308, 174, 788, 425, 913, 163, 215, 505, 283, 654, 802, 484, 4, 726, 987, 244, 374, 155, 331, 165, 401, 148, 586, 937, 195, 899, 224, 597, 572, 753, 472, 891, 549, 170, 921, 150, 222, 10, 122, 905, 557, 458, 555, 356, 338, 339, 755, 651, 717, 591, 857, 583, 159, 767, 979, 19, 606, 380, 818, 616, 569, 285, 988, 59, 986, 794, 843, 468, 402, 790, 877, 453, 879, 256, 548, 678, 41, 580, 135, 923, 815, 415, 596, 385, 14, 610, 189, 517, 112, 82, 392, 530, 352, 689, 103, 573, 341, 592, 306, 149, 629, 261, 373, 868, 854, 485, 784, 743, 437, 617, 603, 60, 18, 571, 97, 169, 37, 34, 98, 273, 759, 151, 566, 108, 723, 677, 147, 367, 709, 660, 668, 281, 358, 303, 202, 221, 845, 852, 785, 718, 985, 746, 473, 210, 251, 157, 365, 544, 959, 233, 856, 263, 410, 42, 581, 160, 411, 257, 213, 693, 882, 208, 683, 301, 846, 940, 564, 212, 190, 960, 181, 716, 952, 849, 936, 537, 276, 232, 957, 685, 346, 619, 294, 287, 883, 782, 713, 996, 841, 467, 972, 656, 109, 556, 932, 386, 634, 152, 205, 114, 329, 446, 35, 81, 246, 482, 461, 420, 130, 796, 744, 844, 90, 432, 969, 73, 322, 956, 894, 334, 834, 666, 773, 735, 831, 602, 258, 500, 659, 980, 47, 760, 28, 463, 728, 26, 438, 703, 585, 870, 803, 361, 139, 533, 589, 307, 800, 344, 620, 912, 413, 105, 372, 866, 635, 125, 193, 486, 918, 16, 973, 313, 154, 820, 954, 769, 640, 11, 32, 87, 588, 262, 832, 396, 892, 914, 171, 901, 715, 403, 999, 417, 614, 58, 945, 705, 692, 259, 414, 593, 539, 354, 464, 239, 297, 70, 179, 576, 115, 183, 318, 71, 730, 327, 860, 300, 509, 284, 518, 177, 887, 188, 330, 859, 874, 911, 180, 653, 896, 720, 858, 350, 776, 421, 819, 370, 783, 89, 381, 748, 33, 536, 405, 568, 762, 138, 56, 853, 113, 546, 810, 460, 194, 871, 440, 661, 265, 756, 176, 824, 806, 529, 298, 24, 110, 454, 679, 873, 699, 333, 293, 7, 133, 550, 513, 118, 827, 623, 353, 770, 662, 102, 245, 542, 669, 976, 455, 168, 470, 989, 766, 511, 672, 65, 704, 120, 314, 821, 319, 377, 535, 642, 682, 68, 875, 431, 684, 953, 270, 855, 321, 927, 225, 638, 884, 497, 248, 459, 267, 897, 219, 328, 423, 75, 310, 978, 426, 551, 792, 889, 920, 734, 578, 457, 837, 304, 909, 885, 282, 641, 671, 478, 292, 949, 407, 881, 779, 227, 902, 637, 448, 128, 508, 199, 78, 761, 495, 323, 524, 712, 201, 904, 584, 701, 975, 126, 812, 153, 38, 908, 192, 362, 240, 554, 252, 167, 811, 156, 359, 211, 198, 131, 941, 275, 3, 590, 434, 488, 950, 740, 136, 223, 737, 447, 652, 218, 1, 490, 994, 443, 676, 349, 658, 838, 890, 630, 85, 922, 646, 612, 422, 696, 104, 919, 51, 99, 521, 394, 598, 250, 835, 465, 101, 842, 816, 714, 449, 655, 925, 558, 436, 496, 708, 119, 817, 77, 964, 264, 340, 236, 17, 624, 76, 84, 441, 238, 745, 268, 872, 129, 325, 829, 53, 736, 186, 741, 930, 504, 378, 203, 124, 575, 433, 786, 80, 462, 552, 944, 395, 288, 337, 506, 175, 600, 828, 825, 962, 797, 977, 739, 907, 955, 207, 742, 547, 412, 280, 869, 67, 260, 648, 886, 383, 695, 608, 360, 48, 963, 200, 140, 898, 931, 958, 966, 710, 502, 916, 900, 384, 299, 780, 162, 942, 514, 670, 673, 191, 809, 798, 79, 917, 451, 733, 439, 910, 515]:
        # if idx >= 714:
            try:
                print(f"Processing sample {idx}/{len(QA_df)}")
                
                # Read video frames
                video_path = os.path.join('UrbanVideo-Bench/videos', str(row['video_id']))
                frames = read_video_frames(video_path)
                
                # Process with TextGrad
                result = textgrad.process(
                    question=row['question'],
                    frames=frames,
                    video_path=video_path,
                    question_category=row['question_category']
                )
                
                # Save result
                output_path = os.path.join(results_dir, f'sample_{idx}.json')
                with open(output_path, 'w') as f:
                    json.dump({
                        'video_id': row['video_id'],
                        'question': row['question'],
                        'question_category': row['question_category'],
                        'ground_truth': row['answer'],
                        'textgrad_result': result
                    }, f, indent=2)
                
                print(f"Saved result to {output_path}")
            except Exception as e:
                print(f"Error processing sample {idx}/{len(QA_df)}: {e}")

if __name__ == "__main__":
    main() 