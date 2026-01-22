import torch
import argparse
import os
from PIL import Image

# 尝试导入特定 Pipeline
# 如果报错，请运行: pip install git+https://github.com/huggingface/diffusers
try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    print("错误: 无法导入 QwenImageEditPlusPipeline。")
    print("请确保你安装了最新版的 diffusers (需要包含 QwenImageEdit 支持的版本)。")
    print("尝试运行: pip install git+https://github.com/huggingface/diffusers")
    import sys
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="使用 Qwen-Image-Edit-2511 和 Gaussian Splash LoRA 修复渲染图像")
    parser.add_argument("--input", "-i", type=str, default=None, help="单张模式：需要修复的渲染图片路径")
    parser.add_argument("--ref", "-r", type=str, default=None, help="单张模式：参考场景图片路径")
    parser.add_argument("--output", "-o", type=str, default="output_fixed.png", help="单张模式：输出图片保存路径")
    
    parser.add_argument("--batch_dir", "-b", type=str, default=None, help="批量模式：包含子文件夹的目录 (例如 fix_pairs_1000)，每个子文件夹应包含 input.png 和 ref.png/jpg")
    
    parser.add_argument("--prompt", type=str, default="高斯泼溅,参考图2的场景图，修复图1的场景图透视并修复空白区域", help="提示词")
    parser.add_argument("--steps", type=int, default=40, help="推理步数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    
    args = parser.parse_args()

    # 检查模式
    mode = "single"
    if args.batch_dir:
        if not os.path.exists(args.batch_dir):
            print(f"错误: 找不到批量目录 {args.batch_dir}")
            return
        mode = "batch"
        print(f"模式: 批量处理 -> {args.batch_dir}")
    else:
        if not args.input or not args.ref:
            # 使用默认值或者报错
            # 这里为了兼容以前的默认参数，如果没有提供batch且没有input/ref，但有默认值的情况(argparse default)，逻辑需要小心
            # 但 argparse default 已经有了。如果用户完全没传参，就会用默认路径。
            # 为了严谨，我们检查 file existence
             if not os.path.exists(args.input) or not os.path.exists(args.ref):
                 print("错误: 请提供 --batch_dir 进行批量处理，或提供有效的 --input 和 --ref 进行单张处理")
                 return
        mode = "single"
        print(f"模式: 单张处理 -> {args.output}")

    # 1. 加载基础模型
    print(">>> 正在加载基础模型 Qwen/Qwen-Image-Edit-2511 ...")
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        pipeline.to("cuda")
    except Exception as e:
        print(f"加载基础模型失败: {e}")
        print("提示: 确保已登录 Hugging Face (huggingface-cli login) 或是网络连接正常。")
        return

    # 2. 加载 LoRA
    lora_id = "dx8152/Qwen-Image-Edit-2511-Gaussian-Splash"
    print(f">>> 正在加载 LoRA: {lora_id} ...")
    try:
        pipeline.load_lora_weights(lora_id)
    except Exception as e:
        print(f"加载 LoRA 失败: {e}")
        return

    print(">>> 模型加载完毕")

    # 定义处理单张图的逻辑
    def process_pair(input_path, ref_path, output_path, seed):
        try:
            image_fix = Image.open(input_path).convert("RGB")
            image_ref = Image.open(ref_path).convert("RGB")
        except Exception as e:
            print(f"  [Error] 读取图片失败 ({input_path}): {e}")
            return

        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)

        inputs = {
            "image": [image_fix, image_ref], 
            "prompt": args.prompt,
            "true_cfg_scale": 4.0, 
            "guidance_scale": 1.0, 
            "num_inference_steps": args.steps,
            "negative_prompt": " ",
            "generator": generator,
            "num_images_per_prompt": 1,
        }

        try:
            with torch.inference_mode():
                output = pipeline(**inputs)
                output_image = output.images[0]
            output_image.save(output_path)
            print(f"  [Success] Saved to {output_path}")
        except Exception as e:
            print(f"  [Error] 推理失败: {e}")

    # 分发任务
    if mode == "single":
        print(f">>> 开始单张处理...")
        process_pair(args.input, args.ref, args.output, args.seed)
        
    elif mode == "batch":
        print(f">>> 开始批量处理: {args.batch_dir}")
        subdirs = sorted([d for d in os.listdir(args.batch_dir) if os.path.isdir(os.path.join(args.batch_dir, d))])
        
        from tqdm import tqdm
        for subdir in tqdm(subdirs, desc="Processing batches"):
            current_dir = os.path.join(args.batch_dir, subdir)
            
            # 寻找 input.png
            input_p = os.path.join(current_dir, "input.png")
            
            # 寻找 ref (可能是 png, jpg, jpeg)
            ref_ups = [f for f in os.listdir(current_dir) if f.startswith("ref.") and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not ref_ups:
                print(f"  [Skip] {subdir}: 找不到 ref 图片")
                continue
            ref_p = os.path.join(current_dir, ref_ups[0])
            
            if not os.path.exists(input_p):
                print(f"  [Skip] {subdir}: 找不到 input.png")
                continue
                
            output_p = os.path.join(current_dir, "output_fixed.png")
            
            # 如果已经存在，可以选择跳过
            # if os.path.exists(output_p):
            #     continue
                
            process_pair(input_p, ref_p, output_p, args.seed)

    print(f">>> 所有任务处理完成！")

if __name__ == "__main__":
    main()
