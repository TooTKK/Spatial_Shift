"""
Test smart floor removal functionality
"""
import sys
from pathlib import Path

# Test clicking on furniture to see if floor is automatically removed
if __name__ == "__main__":
    from sam import SAM2Handler
    
    print("=== Test smart floor removal functionality ===\n")
    
    # Initialize SAM
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    if not Path(checkpoint).exists():
        print(f"❌ Model file not found: {checkpoint}")
        sys.exit(1)
    
    sam = SAM2Handler(checkpoint, config)
    
    # Ask user for test image and coordinates
    print("Please provide test parameters:")
    image_path = input("Image path (e.g., uploads/test.jpg): ").strip()
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    x = int(input("Click X coordinate on furniture: "))
    y = int(input("Click Y coordinate on furniture: "))
    
    output_path = "output/furniture/test_floor_removal.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 Starting segmentation...")
    print(f"   Image: {image_path}")
    print(f"   Coordinates: ({x}, {y})")
    
    # Execute smart segmentation (automatically detects and removes floor)
    result_path, score, related_count = sam.process_segmentation_smart(
        image_path, x, y, output_path
    )
    
    print(f"\n✅ Segmentation complete!")
    print(f"   Output: {result_path}")
    print(f"   Confidence: {score:.3f}")
    print(f"   Related objects: {related_count} items")
    print(f"\nCheck the output image to see if the floor was correctly removed 👀")
