"""
测试本地 LaMa Inpainting（真正的智能补全）
用法：python test_lama.py <图片路径> <x坐标> <y坐标>
例如：python test_lama.py image.png 520 341
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../sam2"))

import numpy as np
from sam import SAM2Handler
from inpainting_local import LocalLamaInpainter

def test_lama_pipeline(image_path, x, y):
    """测试 SAM + 本地 LaMa 完整流程"""
    
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return
    
    print("=" * 70)
    print("🚀 开始测试本地 LaMa Inpainting（真正的智能补全）")
    print("=" * 70)
    print(f"📷 输入图片: {image_path}")
    print(f"📍 点击坐标: ({x}, {y})")
    print()
    
    try:
        # ==================== 步骤 1: SAM 分割 ====================
        print("【步骤 1/3】SAM 分割物体...")
        print("-" * 70)
        
        CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
        MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        sam_engine = SAM2Handler(CHECKPOINT, MODEL_CONFIG)
        
        # 执行分割
        from PIL import Image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        
        sam_engine.predictor.set_image(image_np)
        
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, scores, _ = sam_engine.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        mask = masks[0]
        score = float(scores[0])
        
        print(f"✅ SAM 分割完成！置信度: {score:.4f}")
        print()
        
        # 保存家具切片
        os.makedirs("output/furniture", exist_ok=True)
        furniture_path = f"output/furniture/cutout_{os.path.basename(image_path)}"
        original_rgba = Image.open(image_path).convert("RGBA")
        mask_alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_rgba.size)
        original_rgba.putalpha(mask_alpha)
        original_rgba.save(furniture_path)
        print(f"📦 家具切片已保存: {furniture_path}")
        print()
        
        # ==================== 步骤 2: LaMa Inpainting ====================
        print("【步骤 2/3】本地 LaMa Inpainting（智能补全背景）...")
        print("-" * 70)
        
        lama_engine = LocalLamaInpainter()
        result = lama_engine.process_full_pipeline(
            image_path=image_path,
            mask_array=mask,
            output_dir="output"
        )
        
        print()
        
        # ==================== 步骤 3: 输出结果 ====================
        print("【步骤 3/3】测试完成！")
        print("=" * 70)
        print("✅ 所有步骤成功完成！")
        print("=" * 70)
        print()
        print("📂 输出文件：")
        print(f"  1️⃣  家具切片（透明PNG）: {furniture_path}")
        print(f"  2️⃣  干净背景（智能补全）:  {result['background_clean']}")
        print(f"  3️⃣  遮罩文件:            {result['mask']}")
        print()
        print("💡 效果对比：")
        print("   - 打开原图和背景图对比")
        print("   - 背景应该保持原有风格，只是家具消失了")
        print("   - 不应该是完全不同的房间")
        print()
        
    except Exception as e:
        print("=" * 70)
        print("❌ 测试失败！")
        print("=" * 70)
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python test_lama.py <图片路径> <x坐标> <y坐标>")
        print("例如: python test_lama.py image.png 520 341")
        print()
        print("说明: 使用本地 LaMa 进行真正的智能补全")
        sys.exit(1)
    
    image_path = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    
    test_lama_pipeline(image_path, x, y)
