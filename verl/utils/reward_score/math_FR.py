import re
from typing import Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from mathruler.grader import extract_boxed_content, grade_answer
from rouge import Rouge

pneumonia = {
    "normal": "Clear lung fields, symmetrical bilateral structures, absence of opacities, distinct vascular markings",
    "pneumonia": "Highlights infection-related abnormalities like opacities (fluid/inflammation), consolidation (dense lung tissue), and pleural involvement"
}

# blood = {"basophil": "Large dark purple cytoplasmic granules that often obscure the bilobed nucleus",
#     "eosinophil": "Distinctive bright orange-red granules with a typically bilobed nucleus",
#     "erythroblast": "Large, round nucleus with dense chromatin and deeply basophilic cytoplasm",
#     "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)": "Larger cells with oval/kidney-shaped nuclei or primary granules, less segmented nuclei",
#     "lymphocyte": "Small round cells with scant pale blue cytoplasm and a dense, spherical nucleus",
#     "monocyte": "Largest WBC with abundant gray-blue cytoplasm and a kidney-shaped/ folded nucleus",
#     "neutrophil": "Pale cytoplasm with fine pink granules and characteristic 3-5 lobed nucleus",
#     "platelet": "Small anucleate cellular fragments with purple granules, often clustered"
# }

# derma = {"actinic keratoses and intraepithelial carcinoma": "Rough, scaly patches/plaques; erythematous or brownish; premalignant solar damage; irregular borders",
#     "basal cell carcinoma": "Pearly nodules with telangiectasia; rolled edges; ulceration; slow-growing locally invasive lesions",
#     "benign keratosis-like lesions": "Waxy stuck-on plaques; tan-brown coloration; hyperkeratotic 'greasy' surface; well-demarcated borders",
#     "dermatofibroma": "Firm dermal nodule; characteristic dimple sign; pink-brown coloration; hyperpigmented periphery",
#     "melanoma": "Asymmetrical pattern; color variegation (tan/black/red); irregular borders; rapid evolutionary changes",
#     "melanocytic nevi": "Circular/papular morphology; uniform pigmentation (tan-brown); smooth borders; stable appearance",
#     "vascular lesions": "Red-purple coloration; macular or papular morphology; blanchable; blood vessel proliferation patterns"
#     }

# oct = {"choroidal neovascularization": "Abnormal blood vessel growth beneath retina, fluid accumulation, structural disruption",
#     "diabetic macular edema": "Retinal thickening with fluid-filled cysts, sponge-like appearance in macular region",
#     "drusen": "Yellowish lipid-protein deposits under retina, irregular retinal pigment epithelium",
#     "normal": "Distinct retinal layer stratification, uniform tissue texture, no pathological features"
#     }

# organa = {"bladder": "Rounded pelvic fluid-filled sac, central location, low-density homogeneous",
#     "femur-left": "Dense cortical bone structure, left hip joint articulation, long axis orientation",
#     "femur-right": "Mirror of left femur, right-sided cortical bone, trochanter landmarks",
#     "heart": "Central mediastinal structure, layered myocardium, variable chamber sizes",
#     "kidney-left": "Bean-shaped retroperitoneal organ, left paravertebral position, corticomedullary differentiation",
#     "kidney-right": "Right-sided renal structure, slightly lower position than left, hilum vessels",
#     "liver": "Large right hypochondrium organ, homogeneous parenchyma, venous branching",
#     "lung-left": "Aerated left hemithorax, bronchovascular markings, cardiac notch",
#     "lung-right": "Right thoracic airspace, three-lobe architecture, diaphragmatic surface",
#     "pancreas": "Retroperitoneal glandular structure, head-neck-body-tail continuity",
#     "spleen": "Ovoid left upper quadrant organ, homogeneous enhancement, rib-adjacent",
#     }

# breast = {"malignant": "Irregular shape, spiculated margins, heterogeneous echotexture",
#     "normal, benign": "Smooth contours, circumscribed margins, homogeneous echotexture",
#     }

# tissue = {"Collecting Duct, Connecting Tubule": "Cuboidal epithelial cells with pale cytoplasm, distinct cell borders, and organized tubular structures",
#     "Distal Convoluted Tubule": "Smaller cells with densely packed nuclei, minimal cytoplasm, and convoluted tubular morphology",
#     "Glomerular endothelial cells": "Flattened fenestrated cells forming capillary walls in glomeruli, with elongated nuclei",
#     "Interstitial endothelial cells": "Thin vascular lining cells outside glomeruli, often associated with peritubular capillaries",
#     "Leukocytes": "Round immune cells with high nuclear-to-cytoplasmic ratio, often showing lobed nuclei",
#     "Podocytes": "Large branching cells with foot processes (pedicels) wrapping around glomerular capillaries",
#     "Proximal Tubule Segments": "Large columnar cells with prominent brush borders (microvilli) and eosinophilic cytoplasm",
#     "Thick Ascending Limb": "Simple cuboidal cells with basal membrane infoldings and mitochondria-rich cytoplasm",
#     }

def math_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def math_acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_bleu(candidate_text: str, reference_text: str) -> float: # 计算bleu4
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
        return sentence_bleu( # rouge-L分数试试
            reference_tokens,
            candidate_tokens,
            smoothing_function=smoothie
        )
    except:
        return 0.0
    
def compute_rouge_l(candidate_text: str, reference_text: str) -> float:
    """
    计算候选文本和参考文本之间的Rouge-L分数。
    """
    if not candidate_text:
        return 0.0
    
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate_text, reference_text)
        return scores[0]['rouge-l']['f']
    except ValueError:
        return 0.0

def math_FR_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    format_reward = math_format_reward(predict_str)
    accuracy_reward = math_acc_reward(predict_str, ground_truth)
    
    feature_reward = 0.0
    answer = extract_boxed_content(predict_str)
    
    # if answer in oct and accuracy_reward == 1.0:
    #     # 提取<think>标签内的内容
    #     think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
    #     think_content = think_match.group(1).strip() if think_match else ""
    #     # 计算BLEU分数
    #     feature_reward = compute_bleu(think_content, oct[answer])
    #     # 截断
    #     if feature_reward > 0.25:
    #         feature_reward = 1
    if answer in pneumonia:
        # 提取<think>标签内的内容
        think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else ""
        # 计算BLEU分数
        feature_reward = compute_bleu(think_content, pneumonia[answer])
        # 截断
        # if feature_reward <= 0.1
        #     feature_reward = 0
    # 改为惩罚试试
    return {
        # "overall": 0.7 * accuracy_reward + 0.1 * format_reward + 0.2 * feature_reward,
        "overall": 0.9 * accuracy_reward + 0.1 * format_reward - 2 * feature_reward, 
        "format": format_reward,
        "accuracy": accuracy_reward,
        "feature": feature_reward
    }