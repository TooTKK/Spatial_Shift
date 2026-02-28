"""
测试 Inpainting 功能
用法：python test_inpainting.py <图片路径> <x坐标> <y坐标>
例如：python test_inpainting.py image.png 520 341

完整流程：SAM 分割 → 生成 mask → Replicate inpainting → 获得干净背景
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../sam2"))

import numpy as np
from sam import SAM2Handler
from inpainting import InpaintingHandler

def test_full_pipeline(image_path, x, y):
    """测试完整的分割 + inpainting 流程"""
    
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return
    
    print("=" * 70)
    print("🚀 开始测试完整 Inpainting 流程")
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
        
        # 创建输出目录
        os.makedirs("output/furniture", exist_ok=True)
        furniture_path = f"output/furniture/cutout_{os.path.basename(image_path)}"
        
        # 执行分割（获取 mask）
        from PIL import Image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        
        # 设置图片
        sam_engine.predictor.set_image(image_np)
        
        # 预测
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, scores, _ = sam_engine.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        mask = masks[0]  # 布尔数组，True=物体，False=背景
        score = float(scores[0])
        
        print(f"✅ SAM 分割完成！置信度: {score:.4f}")
        print()
        
        # 保存透明家具切片
        original_rgba = Image.open(image_path).convert("RGBA")
        mask_alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_rgba.size)
        original_rgba.putalpha(mask_alpha)
        original_rgba.save(furniture_path)
        print(f"📦 家具切片已保存: {furniture_path}")
        print()
        
        # ==================== 步骤 2: Inpainting 消除物体 ====================
        print("【步骤 2/3】Replicate Inpainting 消除物体...")
        print("-" * 70)
        
        inpainting_engine = InpaintingHandler()
        result = inpainting_engine.process_full_pipeline(
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
        print(f"  2️⃣  干净背景（已消除）:  {result['background_clean']}")
        print(f"  3️⃣  遮罩文件:            {result['mask']}")
        print()
        print("💡 下一步：")
        print("   - 打开文件查看效果")
        print("   - 如果满意，可以集成到 main.py 的 API 中")
        print("   - 前端可以用这两个文件实现拖拽功能")
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
        print("用法: python test_inpainting.py <图片路径> <x坐标> <y坐标>")
        print("例如: python test_inpainting.py image.png 520 341")
        print()
        print("说明: 会执行完整流程（SAM分割 + Replicate消除背景）")
        sys.exit(1)
    
    image_path = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    
    test_full_pipeline(image_path, x, y)
