"""
Test complete furniture placement workflow
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"))

from sam import SAM2Handler
from inpainting_cloud import CloudInpainter
from furniture_placement import FurniturePlacer


def test_full_workflow():
    """Test complete workflow: segment → remove → place"""
    
    print("=" * 60)
    print("🚀 Starting complete furniture smart move workflow test")
    print("=" * 60)
    
    # Configuration
    test_image = "./blue.png"  # Replace with your test image
    furniture_click = ( 186, 277)  # Furniture position (need replacement)
    new_position = (457, 281)     # New position (need replacement)
    
    if not os.path.exists(test_image):
        print(f"❌ Test image does not exist: {test_image}")
        print("📝 Please create test_images/ directory and place indoor photos inside")
        return
    
    # Initialize all modules
    print("\n📦 Initializing models...")
    sam = SAM2Handler(
        checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    inpainter = CloudInpainter()
    placer = FurniturePlacer()
    
    # Step 1: SAM smart furniture segmentation (automatically recognize related objects)
    print("\n" + "=" * 60)
    print("📍 Step 1: SAM smart furniture segmentation")
    print("=" * 60)
    furniture_path = "output/furniture/test_furniture.png"
    os.makedirs("output/furniture", exist_ok=True)
    
    result_path, score, related_count = sam.process_segmentation_smart(
        test_image,
        furniture_click[0],
        furniture_click[1],
        furniture_path,
        iou_threshold=0.1  # IoU threshold, controls range of related object recognition
    )
    print(f"✅ Smart recognition complete! Confidence: {score:.2f}")
    print(f"   Additionally identified {related_count} related objects (such as chair legs, cushions, etc.)")
    print(f"   Saved to: {furniture_path}")
    
    # Step 2: Inpainting to remove furniture
    print("\n" + "=" * 60)
    print("🎨 Step 2: Remove furniture, generate clean background")
    print("=" * 60)
    background_path = "output/backgrounds/test_clean_bg.png"
    os.makedirs("output/backgrounds", exist_ok=True)
    
    clean_bg = inpainter.inpaint(
        test_image,
        furniture_path,  # Pass path
        background_path
    )
    print(f"✅ Background repair complete!")
    print(f"   Saved to: {background_path}")
    
    # Step 3: Poisson Blending placement (quick test)
    print("\n" + "=" * 60)
    print("🖼️  Step 3a: Poisson Blending furniture placement")
    print("=" * 60)
    os.makedirs("output/placed", exist_ok=True)
    
    poisson_result = placer.poisson_blend(
        furniture_path,
        background_path,
        new_position[0],
        new_position[1],
        furniture_click[0],  # Original position X
        furniture_click[1],  # Original position Y
        output_path="output/placed/test_poisson.png"
    )
    print(f"✅ Poisson blending complete!")
    print(f"   Saved to: {poisson_result}")
    
    # Step 4: AI Blending placement (optional, requires API token)
    # print("\n" + "=" * 60)
    # print("🤖 Step 3b: AI smart blending furniture placement")
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
    #     print(f"✅ AI blending complete!")
    #     print(f"   Saved to: {ai_result}")
    # else:
    #     print("⚠️  Skipping AI blending (requires REPLICATE_API_TOKEN)")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Test complete! Please check the following output files:")
    print("=" * 60)
    print(f"1. Furniture cutout:     {furniture_path}")
    print(f"2. Clean background:     {background_path}")
    print(f"3. Poisson result:       output/placed/test_poisson.png")
    if placer.replicate_token:
        print(f"4. AI blending result:   output/placed/test_ai.png")
    print("\n💡 Tip: Open these images in a browser to view the results")
    print("=" * 60)


if __name__ == "__main__":
    test_full_workflow()
