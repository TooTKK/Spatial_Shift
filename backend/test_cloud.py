"""
Test cloud-based Inpainting (free, no local model required)
Usage: python test_cloud.py <image_path> <x_coord> <y_coord>
Example: python test_cloud.py image.png 520 341
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../sam2"))

import numpy as np
from sam import SAM2Handler
from inpainting_cloud import CloudInpainter

def test_cloud_pipeline(image_path, x, y):
    """Test complete SAM + cloud-based Inpainting pipeline"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image does not exist: {image_path}")
        return
    
    print("=" * 70)
    print("🚀 Starting cloud-based Inpainting test (free API)")
    print("=" * 70)
    print(f"📷 Input image: {image_path}")
    print(f"📍 Click coordinates: ({x}, {y})")
    print()
    
    try:
        # ==================== Step 1: SAM Segmentation ====================
        print("【Step 1/3】SAM object segmentation...")
        print("-" * 70)
        
        CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
        MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        sam_engine = SAM2Handler(CHECKPOINT, MODEL_CONFIG)
        
        # Use smart recognition (automatically recognize related objects like chair legs, pillows, etc.)
        from PIL import Image
        os.makedirs("output/furniture", exist_ok=True)
        furniture_path = f"output/furniture/cutout_{os.path.basename(image_path)}"
        
        result_path, score, related_count = sam_engine.process_segmentation_smart(
            image_path, x, y, furniture_path, iou_threshold=0.1
        )
        
        print(f"✅ Smart recognition complete!")
        print(f"   Confidence: {score:.4f}")
        print(f"   Additionally identified {related_count} related objects (such as legs, cushions, etc.)")
        print()
        
        # Read generated mask for subsequent inpainting
        furniture_img = Image.open(furniture_path)
        mask = np.array(furniture_img.split()[-1]) > 128
        
        print(f"📦 Furniture cutout saved: {furniture_path}")
        print()
        
        # ==================== Step 2: Cloud-based Inpainting ====================
        print("【Step 2/3】Cloud-based Inpainting (background completion)...")
        print("-" * 70)
        
        cloud_engine = CloudInpainter()
        result = cloud_engine.process_full_pipeline(
            image_path=image_path,
            mask_array=mask,
            output_dir="output"
        )
        
        print()
        
        # ==================== Step 3: Output Results ====================
        print("【Step 3/3】Test complete!")
        print("=" * 70)
        print("✅ All steps completed successfully!")
        print("=" * 70)
        print()
        print("📂 Output files:")
        print(f"  1️⃣  Furniture cutout (transparent PNG): {furniture_path}")
        print(f"  2️⃣  Clean background (cloud-generated):  {result['background_clean']}")
        print(f"  3️⃣  Mask file:                           {result['mask']}")
        print()
        print("💡 Note:")
        print("   - Uses free cloud API, no local model required")
        print("   - If API fails, returns original image (Fallback)")
        print("   - Can implement drag-and-drop and compositing features in frontend")
        print()
        
    except Exception as e:
        print("=" * 70)
        print("❌ Test failed!")
        print("=" * 70)
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test_cloud.py <image_path> <x_coord> <y_coord>")
        print("Example: python test_cloud.py image.png 520 341")
        print()
        print("Note: Uses free cloud API for inpainting")
        sys.exit(1)
    
    image_path = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    
    test_cloud_pipeline(image_path, x, y)
