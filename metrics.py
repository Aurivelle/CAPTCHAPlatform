import os
from typing import List, Callable, Dict, Union
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def compute_accuracy(preds: List[str], labels: List[str]) -> float:
    """
    計算整串 CAPTCHA 完全正確率 (全字對)
    preds, labels: 長度相同的 list
    返回 0~1 之間的正確率
    """
    assert len(preds) == len(labels), "Preds and labels must have same length"
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) if labels else 0.0


def _edit_distance(s1: str, s2: str) -> int:
    """
    計算兩個字串的 Levenshtein 編輯距離
    """
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    # now len(s1) >= len(s2)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current = [i]
        for j, c2 in enumerate(s2, start=1):
            add = previous[j] + 1
            delete = current[j - 1] + 1
            replace = previous[j - 1] + (c1 != c2)
            current.append(min(add, delete, replace))
        previous = current
    return previous[-1]


def compute_CER(preds: List[str], labels: List[str]) -> float:
    """
    計算 Character Error Rate (CER): 編輯距離總和 / 總字符數
    """
    total_edits = 0
    total_chars = 0
    for p, l in zip(preds, labels):
        total_edits += _edit_distance(p, l)
        total_chars += len(l)
    return total_edits / total_chars if total_chars > 0 else 0.0


def evaluate_folder(
    predict_fn: Callable[[Image.Image], str], img_dir: str, label_path: str
) -> Dict[str, float]:
    """
    對單一資料夾評估模型表現，計算 accuracy, CER, 平均 BRISQUE (若可)
    predict_fn: 接收 PIL.Image, 返回預測字串
    img_dir: 圖片資料夾路徑
    label_path: 標註檔路徑 (filename label)
    返回 dict: {'accuracy':..., 'cer':..., 'avg_brisque':...}
    """
    # 讀取 label
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            fn, txt = line.strip().split(maxsplit=1)
            labels[fn] = txt

    preds = []
    gts = []
    brisque_scores = []

    for fn, gt in labels.items():
        path = os.path.join(img_dir, fn)
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        pred = predict_fn(img)
        preds.append(pred)
        gts.append(gt)
        if _HAS_BRISQUE:
            score = compute_brisque_score(img)
            brisque_scores.append(score)

    acc = compute_accuracy(preds, gts)
    cer = compute_CER(preds, gts)
    result = {"accuracy": acc, "cer": cer}
    if brisque_scores:
        result["avg_brisque"] = float(np.mean(brisque_scores))
    return result


def plot_comparison(
    metrics: Dict[str, Dict[str, float]], metric_name: str = "accuracy"
):
    """
    繪製多組資料夾的指標比較
    metrics: {label: {metric_name: value, ...}, ...}
    """
    labels = list(metrics.keys())
    values = [metrics[l][metric_name] for l in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.ylabel(metric_name)
    plt.title(f"Comparison of {metric_name}")
    plt.ylim(0, 1 if metric_name != "avg_brisque" else None)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
