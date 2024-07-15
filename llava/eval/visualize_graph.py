import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def load_jsonl(path, num):
    scores = defaultdict(list)
    for i, line in enumerate(open(path)):
        if i > num:
            break
        d = json.loads(line)
        scores[d["category"]].append(d["tuple"][1])
        scores[d["category"] + "_ref"].append(d["tuple"][0])
    return scores


def load_model_results(model_results, num):
    results = {}
    for model_name, result_path in model_results.items():
        scores = load_jsonl(result_path, num)
        result = {}
        for c, s in scores.items():
            if "ref" not in c:
                # 比較対象とターゲットのスコアの平均値の比率をllava-benchのスコアとする
                rel_score = 100 * np.mean(s) / np.mean(scores[c + "_ref"])
                result[c] = rel_score
        results[model_name] = result
    return results

def plot_result(model_results, min_value, max_value):
    """
    # データの設定
    labels = list(model_results[list(model_results.keys())[0]].keys())
    model_scores = {}
    for model_name, result in model_results.items():
        model_scores[model_name] = [max(0, result[label]) for label in labels]
        model_scores[model_name] += model_scores[model_name][:1]
    """

    labels: list[str] = [
        "JA-VG-VQA-500",
        "JA-VLM-Bench-In-the-Wild",
        "Heron-Bench(Detail)",
        "Heron-Bench(Conv)",
        "Heron-Bench(Complex)"
    ]
    model_scores: dict[str, list[float]] = {
        "Japanese Stable VLM": [0.0, 40.50, 25.15, 51.23, 37.84],
        "EvoVLM-JP-v1-7B": [19.70, 51.25, 50.31, 44.42, 40.47],
        "Heron BLIP": [14.51, 33.26, 49.09, 41.51, 45.72],
        "Heron GIT": [15.18, 37.82, 42.77, 54.20, 43.53],
        #"llava-jp-1.3b-v1.0": [12.69, 44.58, 51.21, 41.05, 45.95],
        "llava-jp-1.3b-v1.1": [13.33, 44.40, 50.00, 51.83, 48.98],
        #"llava-karasu-1.1b-v1.0": [13.23, 44.59, 42.16, 43.79, 40.35],
        #"chat-vector-llava-v1.5-7b-ja": [18.64, 42.23, 53.61, 44.36, 44.48],
        "llava-jp-1.3b-v1.1-llava-jp-instruct-108k": [0.0, 17.07, 50.6, 45.31, 33.24],
    }

    for model_name in model_scores.keys():
        model_scores[model_name] += model_scores[model_name][:1]

    # レーダーチャートを描画するための角度を計算
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # 最初の角度をリストの最後に追加して円を閉じる

    # レーダーチャートの描画
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    for i, (model_name, score) in enumerate(model_scores.items()):
        ax.plot(angles, score, color=colorlist[i % len(colorlist)], linewidth=2, label=model_name)
        ax.fill(angles, score, color=colorlist[i % len(colorlist)], alpha=0.05)

    # グラフの見た目を調整
    # メモリの追加
    yticks = np.linspace(min_value, max_value, num=7)  # min_valueからmax_valueまでを5等分
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(round(ytick, 2)) for ytick in yticks])  # メモリに表示する値（小数点第2位まで）

    #ax.set_yticklabels([])
    ax.set_ylim([min_value, max_value])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()


if __name__ == "__main__":
    model_results = {
        "heron-v1-llava-150k": "eval/total/review-llava-bench-heron-v1-llava-150k.json",
        "heron-v1-llava-620k": "eval/total/review-llava-bench-heron-v1-llava-620k.json",
        "llava-jp-siglip": "eval/total/review-llava-bench-llava-jp-siglip.json"
    }
    num = 150
    plot_result(load_model_results(model_results, num), 0, 60)