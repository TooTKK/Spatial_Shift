"""
Test Inpainting functionality
Usage: python test_inpainting.py <image_path> <x_coord> <y_coord>
Example: python test_inpainting.py image.png 520 341

Complete pipeline: SAM segmentation → generate mask → Replicate inpainting → get clean background
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../sam2"))

import numpy as np
from sam import SAM2Handler
from inpainting import InpaintingHandler

def test_full_pipeline(image_path, x, y):
    """Test complete segmentation + inpainting pipeline"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image does not exist: {image_path}")
        return
    
    print("=" * 70)
    print("🚀 Starting complete Inpainting pipeline test")
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
        
        # Create output directory
        os.makedirs("output/furniture", exist_ok=True)
        furniture_path = f"output/furniture/cutout_{os.path.basename(image_path)}"
        
        # Execute segmentation (get mask)
        from PIL import Image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        
        # 设置图片
        sam_engine.predictor.set_image(image_np)
        
        # Predict
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, scores, _ = sam_engine.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        mask = masks[0]  # Boolean array, True=object, False=background
        score = float(scores[0])
        
        print(f"✅ SAM segmentation complete! Confidence: {score:.4f}")
        print()
        
        # Save transparent furniture cutout
        original_rgba = Image.open(image_path).convert("RGBA")
        mask_alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_rgba.size)
        original_rgba.putalpha(mask_alpha)
        original_rgba.save(furniture_path)
        print(f"📦 Furniture cutout saved: {furniture_path}")
        print()
        
        # ==================== Step 2: Inpainting to remove object ====================
        print("【Step 2/3】Replicate Inpainting to remove object...")
        print("-" * 70)
        
        inpainting_engine = InpaintingHandler()
        result = inpainting_engine.process_full_pipeline(
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
        print(f"  2️⃣  Clean background (removed):       {result['background_clean']}")
        print(f"  3️⃣  Mask file:                         {result['mask']}")
        print()
        print("💡 Next steps:")
        print("   - Open files to check the result")
        print("   - If satisfied, can integrate into main.py API")
        print("   - Frontend can use these files to implement drag functionality")
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
        print("Usage: python test_inpainting.py <image_path> <x_coord> <y_coord>")
        print("Example: python test_inpainting.py image.png 520 341")
        print()
        print("Note: Executes complete pipeline (SAM segmentation + Replicate background removal)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    
    test_full_pipeline(image_path, x, y)
