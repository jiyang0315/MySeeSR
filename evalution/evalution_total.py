import os
import json
from collections import defaultdict
from tqdm import tqdm
from cal_psnr_ssim_lpips import calculate_psnr, calculate_ssim, calculate_lpips
import pyiqa


# 评估指标映射
METRIC_FUNCS = {
    "psnr": calculate_psnr,
    "ssim": calculate_ssim,
    "lpips": calculate_lpips,
    "niqe": pyiqa.create_metric('niqe'),
    "nrqm": pyiqa.create_metric('nrqm'),
    "pi": pyiqa.create_metric('pi'),
    "clipiqa": pyiqa.create_metric('clipiqa'),
    "musiq": pyiqa.create_metric('musiq')
}

# 预定义数据集大小
DATASET_SIZE_MAP = {
    "realworld_testdata": 100,
    "chaoji_new": 21
}

# 配置路径
UPSCALE_FOLDER = "/home/jiyang/jiyang/Projects/image_superresolution/datasets/testdata_output_7"
TRUE_FOLDER = "/home/jiyang/jiyang/Projects/image_superresolution/evalution/true_folfer"
OUTPUT_FILE = "/home/jiyang/jiyang/Projects/image_superresolution/evalution/20251114_invsr_lora_5_6_ck.json"

def evaluate_models(upscale_folder, true_folder):
    """计算超分辨率模型的 PSNR、SSIM、LPIPS、NIQE、NRQM、PI、BRISQUE 指标，并计算均值"""
    json_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    # 遍历数据集
    for dataset_name in tqdm(os.listdir(upscale_folder), desc="Processing Datasets"):
        dataset_path = os.path.join(upscale_folder, dataset_name)  

        if not os.path.isdir(dataset_path):
            continue  # 确保是目录

        for model_name in tqdm(os.listdir(dataset_path), desc=f"Processing Models ({dataset_name})"):
            model_path = os.path.join(dataset_path, model_name)
            
            if not os.path.isdir(model_path):
                continue  # 确保是目录

            for image_name in os.listdir(model_path):
                image_path = os.path.join(model_path, image_name)
                
                true_path = os.path.join(true_folder,dataset_name, image_name)
                
                for metric_name, metric_func in METRIC_FUNCS.items():
                    try:
                        score = None
                        
                        # 如果是 chaoji_new，不计算 lpips 和 musiq
                        if dataset_name == "chaoji_new" and metric_name in ["psnr", "ssim", "lpips"]:
                            continue
                        if dataset_name == "chaoji_new" and metric_name == "musiq":
                            continue
                        
                        if metric_name in ["psnr", "ssim", "lpips"]:
                            if dataset_name != "chaoji_new":
                                score = metric_func(image_path, true_path)
                        else:
                            score = metric_func(image_path).item()
                        
                        if score is not None:
                            json_data[dataset_name][metric_name][model_name] += score
                    except Exception as e:
                        print(f"Error processing {image_path} ({metric_name}): {e}")

    # 计算均值
    return calculate_average(json_data)

def calculate_average(data):
    """计算数据集的均值"""
    for dataset_name, dataset_metrics in data.items():
        dataset_size = DATASET_SIZE_MAP.get(dataset_name)  # 获取数据集大小

        for metric_name, model_scores in dataset_metrics.items():
            for model_name in model_scores:
                data[dataset_name][metric_name][model_name] = round(
                    model_scores[model_name] / dataset_size, 3
                )
    return data


def save_results(data, output_file):
    """保存计算结果为 JSON 文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 确保输出目录存在
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump({k: dict(v) for k, v in data.items()}, json_file, indent=4, ensure_ascii=False)
    print(f"数据已保存到 JSON 文件: {output_file}")


def main():
    """主函数"""
    # 计算指标并直接计算均值
    final_results = evaluate_models(UPSCALE_FOLDER, TRUE_FOLDER)

    # 保存最终均值计算结果
    save_results(final_results, OUTPUT_FILE)


if __name__ == "__main__":
    main()
