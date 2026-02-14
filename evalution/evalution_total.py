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

# 配置路径 - SeeSR测试
UPSCALE_FOLDER = "/home/jiyang/jiyang/Projects/SeeSR/evalution/test_output"
TRUE_FOLDER = "/home/jiyang/jiyang/Projects/SeeSR/preset/datasets/test_datasets/DIV2K/HR"
OUTPUT_FILE = "/home/jiyang/jiyang/Projects/SeeSR/evalution/evaluation_results.json"

def evaluate_models(upscale_folder, true_folder):
    """
    计算超分辨率模型的多种指标
    
    目录结构:
    upscale_folder/
        model_name_1/
            sample00/
                0801.png
                0802.png
                ...
        model_name_2/
            sample00/
                ...
    
    true_folder/
        0801.png
        0802.png
        ...
    """
    json_data = defaultdict(lambda: defaultdict(float))
    model_image_counts = defaultdict(int)

    # 遍历模型文件夹
    model_folders = sorted([d for d in os.listdir(upscale_folder) 
                           if os.path.isdir(os.path.join(upscale_folder, d))])
    
    print(f"找到 {len(model_folders)} 个模型待评估: {model_folders}")
    
    for model_name in tqdm(model_folders, desc="评估模型"):
        model_path = os.path.join(upscale_folder, model_name)
        
        # 检查是否有sample00子文件夹
        sample_folder = os.path.join(model_path, "sample00")
        if os.path.isdir(sample_folder):
            image_folder = sample_folder
        else:
            # 如果没有sample00，直接使用模型文件夹
            image_folder = model_path
        
        # 获取该模型的所有图片
        image_files = sorted([f for f in os.listdir(image_folder) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print(f"警告: {model_name} 中没有找到图片")
            continue
        
        print(f"\n处理 {model_name}: {len(image_files)} 张图片")
        
        # 遍历所有图片
        for image_name in tqdm(image_files, desc=f"  {model_name}", leave=False):
            image_path = os.path.join(image_folder, image_name)
            true_path = os.path.join(true_folder, image_name)
            
            # 检查GT是否存在
            if not os.path.exists(true_path):
                print(f"警告: GT不存在: {true_path}")
                continue
            
            model_image_counts[model_name] += 1
            
            # 计算所有指标
            for metric_name, metric_func in METRIC_FUNCS.items():
                try:
                    score = None
                    
                    if metric_name in ["psnr", "ssim", "lpips"]:
                        # 需要GT的指标
                        score = metric_func(image_path, true_path)
                    else:
                        # 无参考指标
                        score = metric_func(image_path).item()
                    
                    if score is not None:
                        json_data[model_name][metric_name] += score
                        
                except Exception as e:
                    print(f"错误: 处理 {image_path} 的 {metric_name} 时出错: {e}")

    # 计算均值
    return calculate_average(json_data, model_image_counts)

def calculate_average(data, model_image_counts):
    """计算每个模型的指标均值"""
    result = {}
    
    for model_name, metrics in data.items():
        image_count = model_image_counts[model_name]
        
        if image_count == 0:
            print(f"警告: {model_name} 没有有效图片")
            continue
        
        result[model_name] = {}
        for metric_name, total_score in metrics.items():
            avg_score = total_score / image_count
            result[model_name][metric_name] = round(avg_score, 4)
        
        result[model_name]["image_count"] = image_count
    
    return result


def save_results(data, output_file):
    """保存计算结果为 JSON 文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 确保输出目录存在
    
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"评估结果已保存: {output_file}")
    print(f"{'='*60}")
    
    # 打印摘要
    print("\n评估摘要:")
    print(f"{'模型名称':<40} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8} {'NIQE↓':>8}")
    print("-" * 80)
    
    for model_name, metrics in sorted(data.items()):
        psnr = metrics.get('psnr', 0)
        ssim = metrics.get('ssim', 0)
        lpips = metrics.get('lpips', 0)
        niqe = metrics.get('niqe', 0)
        print(f"{model_name:<40} {psnr:>8.4f} {ssim:>8.4f} {lpips:>8.4f} {niqe:>8.4f}")
    
    print("-" * 80)


def main():
    """主函数"""
    # 计算指标并直接计算均值
    final_results = evaluate_models(UPSCALE_FOLDER, TRUE_FOLDER)

    # 保存最终均值计算结果
    save_results(final_results, OUTPUT_FILE)


if __name__ == "__main__":
    main()
