"""
SAM 2 local test script
Usage: python test_sam.py <image_path> <x_coord> <y_coord>
Example: python test_sam.py test_image.jpg 500 300
"""
import sys
import os

# Add sam2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../sam2"))

from sam import SAM2Handler

def test_segmentation(image_path, x, y):
    """Test SAM segmentation functionality"""
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image does not exist: {image_path}")
        return
    
    print("=" * 60)
    print("🚀 Starting SAM 2 segmentation test")
    print("=" * 60)
    print(f"📷 Input image: {image_path}")
    print(f"📍 Click coordinates: ({x}, {y})")
    print()
    
    # Configuration paths
    CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    try:
        # Initialize SAM handler
        print("⌛ Loading SAM 2 model...")
        sam_engine = SAM2Handler(CHECKPOINT, MODEL_CONFIG)
        print("✅ Model loaded successfully!")
        print()
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", f"furniture_{os.path.basename(image_path).rsplit('.', 1)[0]}.png")
        
        # Execute segmentation
        print("⌛ Executing segmentation...")
        result_path, score = sam_engine.process_segmentation(image_path, x, y, output_path)
        
        print("=" * 60)
        print("✅ Segmentation complete!")
        print("=" * 60)
        print(f"📁 Output file: {result_path}")
        print(f"🎯 Confidence score: {score:.4f}")
        print()
        print("💡 Tip: Open the output file to view the segmented furniture (transparent background PNG)")
        
    except Exception as e:
        print("=" * 60)
        print("❌ Test failed!")
        print("=" * 60)
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test_sam.py <image_path> <x_coord> <y_coord>")
        print("Example: python test_sam.py test_image.jpg 500 300")
        print()
        print("Tip: The coordinates should be any point on the object you want to segment")
        sys.exit(1)
    
    image_path = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    
    test_segmentation(image_path, x, y)
