"""
Simplified test script - step-by-step furniture placement test
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
    print("🧪 Simplified test: Smart furniture placement")
    print("=" * 60)
    
    # ========== Configuration section (modify here) ==========
    # Replace with your test image path
    TEST_IMAGE = "test_images/room.jpg"  # 👈 Change to your image path
    
    # Click coordinates (furniture position)
    FURNITURE_X = 400  # 👈 Change to your furniture X coordinate in image
    FURNITURE_Y = 300  # 👈 Change to your furniture Y coordinate in image
    
    # New position coordinates
    NEW_X = 800  # 👈 Change to desired X coordinate
    NEW_Y = 400  # 👈 Change to desired Y coordinate
    # =========================================
    
    # Check if image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"\n❌ Test image does not exist: {TEST_IMAGE}")
        print("\n📝 Quick start steps:")
        print("1. Create test_images directory")
        print("2. Place an indoor photo inside (e.g., room.jpg)")
        print("3. Open with image viewer, record furniture position coordinates")
        print("4. Modify coordinate values in the code above")
        print("\n💡 Or test with existing images:")
        
        # Check if there are other available images
        possible_dirs = [".", "output", "../"]
        for d in possible_dirs:
            if os.path.exists(d):
                images = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    print(f"   Found in {d}/ directory: {', '.join(images[:3])}")
        return
    
    # Create output directories
    for folder in ["output/furniture", "output/backgrounds", "output/placed"]:
        os.makedirs(folder, exist_ok=True)
    
    # Display image information
    img = Image.open(TEST_IMAGE)
    print(f"\n📸 Test image information:")
    print(f"   Path: {TEST_IMAGE}")
    print(f"   Size: {img.width}x{img.height}")
    print(f"   Click coordinates: ({FURNITURE_X}, {FURNITURE_Y})")
    print(f"   New position: ({NEW_X}, {NEW_Y})")
    
    # Initialize models
    print("\n" + "=" * 60)
    print("📦 Step 1: Initialize models...")
    print("=" * 60)
    
    print("Loading SAM2...")
    sam = SAM2Handler(
        checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    print("✅ SAM2 loading complete")
    
    print("Initializing Inpainting...")
    inpainter = CloudInpainter()
    print("✅ Inpainting initialization complete")
    
    print("Initializing Furniture Placer...")
    placer = FurniturePlacer()
    print("✅ Furniture Placer initialization complete")
    
    # Step 2: Segment furniture
    print("\n" + "=" * 60)
    print("📍 Step 2: Segment furniture...")
    print("=" * 60)
    
    furniture_path = "output/furniture/test_furniture.png"
    result_path, score = sam.process_segmentation(
        TEST_IMAGE,
        FURNITURE_X,
        FURNITURE_Y,
        furniture_path
    )
    
    print(f"✅ Segmentation complete!")
    print(f"   Output: {furniture_path}")
    print(f"   Confidence: {score:.3f}")
    
    # Step 3: Remove furniture
    print("\n" + "=" * 60)
    print("🎨 Step 3: Remove furniture, generate clean background...")
    print("=" * 60)
    
    background_path = "output/backgrounds/test_clean_bg.png"
    clean_bg = inpainter.inpaint(
        TEST_IMAGE,
        furniture_path,
        background_path
    )
    
    print(f"✅ Background repair complete!")
    print(f"   Output: {background_path}")
    
    # Step 4: Poisson Blending quick placement
    print("\n" + "=" * 60)
    print("⚡ Step 4: Poisson Blending quick placement...")
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
    
    print(f"✅ Poisson blending complete!")
    print(f"   Output: {poisson_path}")
    
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
