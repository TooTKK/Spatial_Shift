"""
简化版测试脚本 - 一步步测试家具放置
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"))

from sam import SAM2Handler
from inpainting_cloud import CloudInpainter
from furniture_placement import FurniturePlacer
from PIL import Image

def main():
    print("=" * 60)
    print("🧪 简化版测试：家具智能放置")
    print("=" * 60)
    
    # ========== 配置区域（修改这里） ==========
    # 请替换为你的测试图片路径
    TEST_IMAGE = "test_images/room.jpg"  # 👈 改成你的图片路径
    
    # 点击坐标（家具位置）
    FURNITURE_X = 400  # 👈 改成你图片中家具的X坐标
    FURNITURE_Y = 300  # 👈 改成你图片中家具的Y坐标
    
    # 新位置坐标
    NEW_X = 800  # 👈 改成想要移动到的X坐标
    NEW_Y = 400  # 👈 改成想要移动到的Y坐标
    # ==========================================
    
    # 检查图片是否存在
    if not os.path.exists(TEST_IMAGE):
        print(f"\n❌ 测试图片不存在: {TEST_IMAGE}")
        print("\n📝 快速开始步骤：")
        print("1. 创建 test_images 目录")
        print("2. 放入一张室内照片（如 room.jpg）")
        print("3. 用图片查看器打开，记录家具位置的坐标")
        print("4. 修改上面代码中的坐标值")
        print("\n💡 或者使用现有图片测试：")
        
        # 检查是否有其他可用图片
        possible_dirs = [".", "output", "../"]
        for d in possible_dirs:
            if os.path.exists(d):
                images = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    print(f"   在 {d}/ 目录找到: {', '.join(images[:3])}")
        return
    
    # 创建输出目录
    for folder in ["output/furniture", "output/backgrounds", "output/placed"]:
        os.makedirs(folder, exist_ok=True)
    
    # 显示图片信息
    img = Image.open(TEST_IMAGE)
    print(f"\n📸 测试图片信息:")
    print(f"   路径: {TEST_IMAGE}")
    print(f"   尺寸: {img.width}x{img.height}")
    print(f"   点击坐标: ({FURNITURE_X}, {FURNITURE_Y})")
    print(f"   新位置: ({NEW_X}, {NEW_Y})")
    
    # 初始化模型
    print("\n" + "=" * 60)
    print("📦 步骤1: 初始化模型...")
    print("=" * 60)
    
    print("正在加载 SAM2...")
    sam = SAM2Handler(
        checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    print("✅ SAM2 加载完成")
    
    print("正在初始化 Inpainting...")
    inpainter = CloudInpainter()
    print("✅ Inpainting 初始化完成")
    
    print("正在初始化 Furniture Placer...")
    placer = FurniturePlacer()
    print("✅ Furniture Placer 初始化完成")
    
    # 步骤2：分割家具
    print("\n" + "=" * 60)
    print("📍 步骤2: 分割家具...")
    print("=" * 60)
    
    furniture_path = "output/furniture/test_furniture.png"
    result_path, score = sam.process_segmentation(
        TEST_IMAGE,
        FURNITURE_X,
        FURNITURE_Y,
        furniture_path
    )
    
    print(f"✅ 分割完成！")
    print(f"   输出: {furniture_path}")
    print(f"   置信度: {score:.3f}")
    
    # 步骤3：移除家具
    print("\n" + "=" * 60)
    print("🎨 步骤3: 移除家具，生成干净背景...")
    print("=" * 60)
    
    background_path = "output/backgrounds/test_clean_bg.png"
    clean_bg = inpainter.inpaint(
        TEST_IMAGE,
        furniture_path,
        background_path
    )
    
    print(f"✅ 背景修复完成！")
    print(f"   输出: {background_path}")
    
    # 步骤4：Poisson Blending快速放置
    print("\n" + "=" * 60)
    print("⚡ 步骤4: Poisson Blending 快速放置...")
    print("=" * 60)
    
    poisson_path = "output/placed/test_poisson.png"
    poisson_result = placer.poisson_blend(
        furniture_path,
        background_path,
        NEW_X,
        NEW_Y,
        FURNITURE_X,
        FURNITURE_Y,
        output_path=poisson_path
    )
    
    print(f"✅ Poisson融合完成！")
    print(f"   输出: {poisson_path}")
    
    # 步骤5：AI融合（可选）
    print("\n" + "=" * 60)
    print("🤖 步骤5: AI智能融合放置（可选）...")
    print("=" * 60)
    
    if placer.replicate_token:
        print("⏳ 调用Replicate API（预计30秒）...")
        ai_path = "output/placed/test_ai.png"
        
        try:
            ai_result = placer.ai_blend(
                furniture_path,
                background_path,
                NEW_X,
                NEW_Y,
                FURNITURE_X,
                FURNITURE_Y,
                output_path=ai_path
            )
            print(f"✅ AI融合完成！")
            print(f"   输出: {ai_path}")
        except Exception as e:
            print(f"⚠️  AI融合失败: {e}")
            print(f"   已有Poisson结果可用")
    else:
        print("⚠️  跳过AI融合（需要 REPLICATE_API_TOKEN）")
        print("   在 .env 文件中配置后可启用高质量AI融合")
    
    # 完成
    print("\n" + "=" * 60)
    print("🎉 测试完成！请查看输出文件：")
    print("=" * 60)
    print(f"1️⃣  原图:         {TEST_IMAGE}")
    print(f"2️⃣  家具抠图:     {furniture_path}")
    print(f"3️⃣  干净背景:     {background_path}")
    print(f"4️⃣  快速融合:     {poisson_path}")
    if placer.replicate_token and os.path.exists("output/placed/test_ai.png"):
        print(f"5️⃣  AI融合:       output/placed/test_ai.png")
    
    print("\n💡 打开图片对比效果：")
    print(f"   open {poisson_path}")
    if os.path.exists("output/placed/test_ai.png"):
        print(f"   open output/placed/test_ai.png")
    
    print("\n📊 性能对比：")
    print("   Poisson Blending: ~1秒,  基础效果")
    print("   AI Blend:         ~30秒, 完美光照/阴影/透视")
    print("=" * 60)


if __name__ == "__main__":
    main()
