#!/bin/bash
# ============================================================================
# SeeSR 评估脚本 - 评估所有模型在DIV2K测试集上的表现
# ============================================================================

cd /home/jiyang/jiyang/Projects/SeeSR/evalution

echo "开始评估 SeeSR 模型..."
echo "测试输出目录: test_output/"
echo "GT目录: ../preset/datasets/test_datasets/DIV2K/HR"
echo ""

python evalution_total.py

echo ""
echo "评估完成！查看结果: evaluation_results.json"

