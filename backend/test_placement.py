"""
测试家具放置完整流程
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"))

from sam import SAM2Handler
from inpainting_cloud import CloudInpainter
from furniture_placement import FurniturePlacer


def test_full_workflow():
    """测试完整工作流：分割 -> 移除 -> 放置"""
    
    print("=" * 60)
    print("🚀 开始测试家具智能移动完整流程")
    print("=" * 60)
    
    # 配置
    test_image = "./blue.png"  # 替换为你的测试图片
    furniture_click = ( 186, 277)  # 家具位置（需替换）
    new_position = (457, 281)     # 新位置（需替换）
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图片不存在: {test_image}")
        print("📝 请创建 test_images/ 目录并放入室内照片")
        return
    
    # 初始化所有模块
    print("\n📦 初始化模型...")
    sam = SAM2Handler(
        checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    inpainter = CloudInpainter()
    placer = FurniturePlacer()
    
    # 步骤1：SAM智能分割家具（自动识别相关物体）
    print("\n" + "=" * 60)
    print("📍 步骤1: SAM智能分割家具")
    print("=" * 60)
    furniture_path = "output/furniture/test_furniture.png"
    os.makedirs("output/furniture", exist_ok=True)
    
    result_path, score, related_count = sam.process_segmentation_smart(
        test_image,
        furniture_click[0],
        furniture_click[1],
        furniture_path,
        iou_threshold=0.1  # IoU阈值，控制识别相关物体的范围
    )
    print(f"✅ 智能识别完成！置信度: {score:.2f}")
    print(f"   额外识别到 {related_count} 个相关物体（如椅子脚、靠枕等）")
    print(f"   保存至: {furniture_path}")
    
    # 步骤2：Inpainting移除家具
    print("\n" + "=" * 60)
    print("🎨 步骤2: 移除家具，生成干净背景")
    print("=" * 60)
    background_path = "output/backgrounds/test_clean_bg.png"
    os.makedirs("output/backgrounds", exist_ok=True)
    
    clean_bg = inpainter.inpaint(
        test_image,
        furniture_path,  # 传入路径
        background_path
    )
    print(f"✅ 背景修复完成！")
    print(f"   保存至: {background_path}")
    
    # 步骤3：Poisson Blending放置（快速测试）
    print("\n" + "=" * 60)
    print("🖼️  步骤3a: Poisson Blending 放置家具")
    print("=" * 60)
    os.makedirs("output/placed", exist_ok=True)
    
    poisson_result = placer.poisson_blend(
        furniture_path,
        background_path,
        new_position[0],
        new_position[1],
        furniture_click[0],  # 原位置X
        furniture_click[1],  # 原位置Y
        output_path="output/placed/test_poisson.png"
    )
    print(f"✅ Poisson融合完成！")
    print(f"   保存至: {poisson_result}")
    
    # 步骤4：AI Blending放置（可选，需要API token）
    # print("\n" + "=" * 60)
    # print("🤖 步骤3b: AI智能融合放置家具")
    # print("=" * 60)
    
    # if placer.replicate_token:
    #     ai_result = placer.ai_blend(
    #         furniture_path,
    #         background_path,
    #         new_position[0],
    #         new_position[1],
    #         furniture_click[0],
    #         furniture_click[1],
    #         output_path="output/placed/test_ai.png"
    #     )
    #     print(f"✅ AI融合完成！")
    #     print(f"   保存至: {ai_result}")
    # else:
    #     print("⚠️  跳过AI融合（需要REPLICATE_API_TOKEN）")
    
    # 总结
    print("\n" + "=" * 60)
    print("🎉 测试完成！请查看以下输出文件：")
    print("=" * 60)
    print(f"1. 家具抠图:     {furniture_path}")
    print(f"2. 干净背景:     {background_path}")
    print(f"3. Poisson结果:  output/placed/test_poisson.png")
    if placer.replicate_token:
        print(f"4. AI融合结果:   output/placed/test_ai.png")
    print("\n💡 提示：在浏览器中打开这些图片查看效果")
    print("=" * 60)


if __name__ == "__main__":
    test_full_workflow()
