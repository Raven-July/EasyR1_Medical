# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from mathruler.grader import extract_boxed_content, grade_answer, _str_is_int

retina = {"0": "Normal retinal structure with no detectable pathological changes or microaneurysms",
    "1": "Isolated microaneurysms only, without hemorrhages or exudates",
    "2": "Multiple microaneurysms, dot/blot hemorrhages, and possible hard exudates",
    "3": "Extensive intraretinal hemorrhages, venous beading, or intraretinal microvascular abnormalities (IRMA)",
    "4": "Neovascularization (optic disc/new vessels elsewhere), vitreous/preretinal hemorrhage, or fibrous proliferation"
    }

def math_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def regress_acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    if _str_is_int(answer) and 0 <= int(float(answer)) <= 4:
        if grade_answer(answer, ground_truth):
            return 1.0
        elif abs(int(float(answer)) - int(ground_truth)) == 1:
            return 0.25
        elif abs(int(float(answer)) - int(ground_truth)) == 2:
            return 0.0625
        else:
            return 0.0
    return 0.0

def compute_bleu(candidate_text: str, reference_text: str) -> float:
    """
    计算候选文本和参考文本之间的BLEU分数。
    """
    if not candidate_text:
        return 0.0
    
    reference_tokens = [reference_text.split()]
    candidate_tokens = candidate_text.split()
    
    if not candidate_tokens:
        return 0.0
    
    smoothie = SmoothingFunction().method1
    try:
        return sentence_bleu(
            reference_tokens,
            candidate_tokens,
            smoothing_function=smoothie
        )
    except:
        return 0.0


def regress_FR_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    format_reward = math_format_reward(predict_str)
    accuracy_reward = regress_acc_reward(predict_str, ground_truth)
    if accuracy_reward == 1.0:
        acc = 1.0
    else:
        acc = 0.0

    feature_reward = 0.0
    answer = extract_boxed_content(predict_str)
    
    # if answer in retina and acc == 1.0:
    #     # 提取<think>标签内的内容
    #     think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
    #     think_content = think_match.group(1).strip() if think_match else ""
    #     # 计算BLEU分数
    #     feature_reward = compute_bleu(think_content, retina[answer])
    #     # 截断
    #     if feature_reward > 0.25:
    #         feature_reward = 1
    if answer in retina:
        # 提取<think>标签内的内容
        think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else ""
        # 计算BLEU分数
        feature_reward = compute_bleu(think_content, retina[answer])
    
    return {
        # "overall": 0.7 * accuracy_reward + 0.1 * format_reward + 0.2 * feature_reward,
        "overall": 0.9 * accuracy_reward + 0.1 * format_reward - 2 * feature_reward, 
        "format": format_reward,
        "accuracy": acc,
        "feature": feature_reward
    }