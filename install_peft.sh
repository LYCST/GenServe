#!/bin/bash

# PEFT 安装脚本
# 用于安装LoRA功能所需的PEFT依赖

echo "🔧 安装 PEFT 依赖..."
echo "===================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

echo "📦 安装 PEFT 库..."
pip install peft>=0.7.0

if [ $? -eq 0 ]; then
    echo "✅ PEFT 安装成功"
else
    echo "❌ PEFT 安装失败"
    exit 1
fi

echo "📦 安装 Transformers PEFT 支持..."
pip install transformers>=4.35.0

if [ $? -eq 0 ]; then
    echo "✅ Transformers PEFT 支持安装成功"
else
    echo "❌ Transformers PEFT 支持安装失败"
    exit 1
fi

echo "🔍 验证安装..."
python3 -c "
try:
    from peft import PeftModel
    print('✅ PEFT 导入成功')
except ImportError as e:
    print(f'❌ PEFT 导入失败: {e}')
    exit(1)

# 注意：PeftConfig在新版本transformers中可能不可用，我们只需要PEFT库本身
print('✅ PEFT 依赖验证通过')

print('🎉 PEFT 依赖安装完成！')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ PEFT 依赖安装完成！"
    echo "📋 现在可以使用 LoRA 功能了"
    echo ""
    echo "💡 使用提示:"
    echo "1. 确保 LoRA 文件(.safetensors)已放置在配置的路径中"
    echo "2. 使用 /loras 接口获取可用 LoRA 列表"
    echo "3. 在生成请求中通过 loras 参数指定 LoRA 和权重"
else
    echo "❌ PEFT 依赖验证失败"
    exit 1
fi 