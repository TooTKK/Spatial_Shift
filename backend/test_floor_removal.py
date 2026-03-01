"""
测试智能地板移除功能
"""
import sys
from pathlib import Path

# 测试点击家具，看是否会自动移除地板
if __name__ == "__main__":
    from sam import SAM2Handler
    
    print("=== 测试智能地板移除功能 ===\n")
    
    # 初始化 SAM
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    if not Path(checkpoint).exists():
        print(f"❌ 找不到模型文件: {checkpoint}")
        sys.exit(1)
    
    sam = SAM2Handler(checkpoint, config)
    
    # 请用户提供测试图片和坐标
    print("请提供测试参数：")
    image_path = input("图片路径 (例如: uploads/test.jpg): ").strip()
    
    if not Path(image_path).exists():
        print(f"❌ 找不到图片: {image_path}")
        sys.exit(1)
    
    x = int(input("点击家具的 X 坐标: "))
    y = int(input("点击家具的 Y 坐标: "))
    
    output_path = "output/furniture/test_floor_removal.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 开始分割...")
    print(f"   图片: {image_path}")
    print(f"   坐标: ({x}, {y})")
    
    # 执行智能分割（会自动检测并移除地板）
    result_path, score, related_count = sam.process_segmentation_smart(
        image_path, x, y, output_path
    )
    
    print(f"\n✅ 分割完成！")
    print(f"   输出: {result_path}")
    print(f"   置信度: {score:.3f}")
    print(f"   相关物体: {related_count} 个")
    print(f"\n检查输出图片，看地板是否被正确移除 👀")
