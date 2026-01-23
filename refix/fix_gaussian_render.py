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
    parser.add_argument("--steps", type=int, default=25, help="推理步数")
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
    
    # [Optimization] 开启 TF32 加速 (针对 Ampere架构及以上 GPU: RTX 30/40, A100等)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        pipeline.to("cuda")
        
        # [Optimization] 尝试编译模型以加速 (首次运行会慢几分钟进行编译，之后会变快)
        # 如果遇到兼容性问题，可以注释掉下面这行
        # print(">>> [Info] Compiling UNet/Transformer for speedup...")
        # pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead", fullgraph=True)
        
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
    def process_pair(input_path, ref_paths, output_path, seed):
        try:
            image_fix = Image.open(input_path).convert("RGB")
            
            # 支持单个路径或列表
            if isinstance(ref_paths, str):
                ref_paths = [ref_paths]
                
            image_refs = []
            for rp in ref_paths:
                image_refs.append(Image.open(rp).convert("RGB"))
                
        except Exception as e:
            print(f"  [Error] 读取图片失败 ({input_path}): {e}")
            return

        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)

        # 构造输入: [Input, Ref1, Ref2, ...]
        input_images = [image_fix] + image_refs
        
        inputs = {
            "image": input_images, 
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
                # [Optimization] 支持批量推理 (虽然输入是单组 [Fix, Ref])
                # 如果未来需要真正的 batching (一次修多张图)，需要在这里重新组织 inputs
                # 目前主要加速点在于 compile 和 pipeline 优化
                output = pipeline(**inputs)
                output_image = output.images[0]
            output_image.save(output_path)
            print(f"  [Success] Saved to {output_path} (Refs: {len(image_refs)})")
        except Exception as e:
            print(f"  [Error] 推理失败: {e}")

    # 分发任务
    if mode == "single":
        print(f">>> 开始单张处理...")
        process_pair(args.input, args.ref, args.output, args.seed)
        
    elif mode == "batch":
        print(f">>> 开始批量处理: {args.batch_dir}")
        subdirs = [d for d in os.listdir(args.batch_dir) if os.path.isdir(os.path.join(args.batch_dir, d))]
        subdirs.sort(key=lambda x: int(x) if x.isdigit() else x)
        
        # [Optimization] 数据集准备: 提前扫描所有任务
        tasks = []
        for subdir in subdirs:
            current_dir = os.path.join(args.batch_dir, subdir)
            input_p = os.path.join(current_dir, "input.png")
            
            ref_candidates = [f for f in os.listdir(current_dir) if (f.startswith("ref_") or f.startswith("ref.")) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            ref_candidates.sort()
            
            output_p = os.path.join(current_dir, "output_fixed.png")
            if os.path.exists(output_p):
                print(f"  [Skip] {subdir}: output_fixed.png 已存在")
                continue
                
            if not ref_candidates or not os.path.exists(input_p):
                continue
                
            ref_paths = [os.path.join(current_dir, rc) for rc in ref_candidates]
            tasks.append({
                'input': input_p,
                'refs': ref_paths,
                'output': output_p,
                'id': subdir
            })
            
        print(f"[Info] 共收集到 {len(tasks)} 个待处理任务。")
        
        # [Optimization] 简单的 Batching 策略:
        # 由于每组任务的 Ref 数量可能不同(虽然现在都是1张)，且图片尺寸可能微调，
        # 直接由 PyTorch 进行 batching 比较困难且容易 OOM。
        # 这里保持 Loop 但优化数据加载部分。
        
        from tqdm import tqdm
        for task in tqdm(tasks, desc="Processing batches"):
            process_pair(task['input'], task['refs'], task['output'], args.seed)

    print(f">>> 所有任务处理完成！")

if __name__ == "__main__":
    main()
