"""
Intelligent Furniture Placement Module
Supports two approaches:
1. Poisson Blending (fast prototype)
2. Stable Diffusion (intelligent blending)
"""
import os
import cv2
import numpy as np
from PIL import Image
import requests
import io
from dotenv import load_dotenv

load_dotenv()

class FurniturePlacer:
    def __init__(self):
        """Initialize furniture placer"""
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if self.replicate_token:
            print("✅ Furniture placer initialized successfully (supports AI blending)")
        else:
            print("⚠️  REPLICATE_API_TOKEN not found, only supports basic blending")
    
    def estimate_scale(self, original_y, new_y, image_height):
        """
        Estimate depth scaling based on Y coordinate
        Principle: In indoor scenes, larger Y (lower) typically indicates closer, objects should be larger
        """
        # Simplified perspective scaling: Linear interpolation based on Y coordinate
        # Assume image bottom is foreground (scale=1.2), top is background (scale=0.5)
        original_ratio = original_y / image_height
        new_ratio = new_y / image_height
        
        # Base scaling factor (between 0.5 and 1.5)
        original_scale = 0.5 + original_ratio * 1.0
        new_scale = 0.5 + new_ratio * 1.0
        
        # Relative scaling
        relative_scale = new_scale / original_scale
        return np.clip(relative_scale, 0.3, 3.0)  # Limit scaling range
    
    def poisson_blend(self, furniture_img_path, background_img_path, x, y, 
                      original_x=None, original_y=None, output_path="output/placed.png"):
        """
        Simplified version: Directly use Alpha blending for furniture placement (most stable)
        
        Args:
            furniture_img_path: Transparent PNG furniture cutout path (original image size, chair at original position, rest transparent)
            background_img_path: Clean background image path
            x, y: New position center coordinates
            original_x, original_y: Original position coordinates (for scale estimation)
            output_path: Output path
        
        Returns:
            Output image path
        """
        print("⏳ Using Alpha Blending for furniture placement...")
        
        # Read images
        furniture_full = Image.open(furniture_img_path).convert("RGBA")
        background = Image.open(background_img_path).convert("RGB")
        
        # 1. First crop actual furniture bounding box from original-size PNG
        furniture_np = np.array(furniture_full)
        alpha_channel = furniture_np[:, :, 3]
        
        # Find boundaries of non-transparent area
        rows = np.any(alpha_channel > 0, axis=1)
        cols = np.any(alpha_channel > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            print("   ⚠️  No non-transparent area found")
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Crop out furniture part
        furniture = furniture_full.crop((x_min, y_min, x_max + 1, y_max + 1))
        print(f"   ✂️  Cropping furniture from original image: original position({x_min}, {y_min}), size{furniture.width}x{furniture.height}")
        
        # Auto scaling: Estimate depth based on Y coordinate
        if original_x and original_y:
            scale = self.estimate_scale(original_y, y, background.height)
            new_width = int(furniture.width * scale)
            new_height = int(furniture.height * scale)
            furniture = furniture.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"   📏 Auto scaling: {scale:.2f}x (original Y={original_y}, new Y={y})")
        
        # Convert to numpy array
        furniture_np = np.array(furniture)
        background_np = np.array(background)
        
        # Extract RGB and Alpha channels
        furniture_rgb = furniture_np[:, :, :3]
        furniture_alpha = furniture_np[:, :, 3:4] / 255.0  # Normalize to 0-1
        
        # Calculate placement position (center aligned)
        center_x = int(x)
        center_y = int(y)
        half_w = furniture.width // 2
        half_h = furniture.height // 2
        
        # Calculate target area range
        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = x1 + furniture.width
        y2 = y1 + furniture.height
        
        # Handle boundary cases: Calculate valid range
        bg_h, bg_w = background_np.shape[:2]
        
        # Valid region on background
        bg_x1 = max(0, x1)
        bg_y1 = max(0, y1)
        bg_x2 = min(bg_w, x2)
        bg_y2 = min(bg_h, y2)
        
        # Corresponding valid region on furniture
        furn_x1 = bg_x1 - x1
        furn_y1 = bg_y1 - y1
        furn_x2 = furn_x1 + (bg_x2 - bg_x1)
        furn_y2 = furn_y1 + (bg_y2 - bg_y1)
        
        # Check if there's valid region
        if bg_x2 <= bg_x1 or bg_y2 <= bg_y1:
            print("   ⚠️  Furniture completely exceeds background boundaries")
            result = Image.fromarray(background_np)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.save(output_path)
            return output_path
        
        # Extract corresponding regions
        bg_roi = background_np[bg_y1:bg_y2, bg_x1:bg_x2]
        furn_rgb_roi = furniture_rgb[furn_y1:furn_y2, furn_x1:furn_x2]
        furn_alpha_roi = furniture_alpha[furn_y1:furn_y2, furn_x1:furn_x2]
        
        # Alpha blending: Preserve all details (including semi-transparent chair legs)
        # result = foreground * alpha + background * (1 - alpha)
        blended_roi = (furn_rgb_roi * furn_alpha_roi + 
                       bg_roi * (1 - furn_alpha_roi)).astype(np.uint8)
        
        # Put blended result back to background
        result_np = background_np.copy()
        result_np[bg_y1:bg_y2, bg_x1:bg_x2] = blended_roi
        
        # Save result
        result = Image.fromarray(result_np)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path)
        
        print(f"   ✅ Furniture placement successful: {output_path}")
        return output_path
    
    def ai_blend(self, furniture_img_path, background_img_path, x, y, 
                 original_x=None, original_y=None, output_path="output/placed_ai.png"):
        """
        Approach A: Use Stable Diffusion for intelligent blending
        
        Principle: Create mask at new position, let AI redraw with furniture image + background, automatically handle perspective and lighting
        
        Args:
            furniture_img_path: Transparent PNG furniture cutout path
            background_img_path: Clean background image path
            x, y: New position center coordinates
            original_x, original_y: Original position coordinates (for scale estimation)
            output_path: Output path
        
        Returns:
            Output image path
        """
        if not self.replicate_token:
            print("❌ Need REPLICATE_API_TOKEN to use AI blending")
            return None
        
        print("⏳ Using Stable Diffusion for intelligent furniture placement...")
        
        # Read images
        furniture = Image.open(furniture_img_path).convert("RGBA")
        background = Image.open(background_img_path).convert("RGB")
        
        # Estimate scaling (consistent with Poisson approach)
        if original_x and original_y:
            scale = self.estimate_scale(original_y, y, background.height)
            new_width = int(furniture.width * scale)
            new_height = int(furniture.height * scale)
            furniture = furniture.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"   📏 Pre-scaling: {scale:.2f}x")
        
        # First use Alpha blending to create preliminary composite (as input for SD)
        furniture_rgb = np.array(furniture.convert("RGB"))
        furniture_alpha = np.array(furniture.split()[-1])
        background_np = np.array(background)
        
        # Calculate placement position
        half_w = furniture.width // 2
        half_h = furniture.height // 2
        x1 = max(0, x - half_w)
        y1 = max(0, y - half_h)
        x2 = min(background.width, x + half_w)
        y2 = min(background.height, y + half_h)
        
        # Crop furniture
        furniture_x1 = half_w - (x - x1)
        furniture_y1 = half_h - (y - y1)
        furniture_x2 = furniture_x1 + (x2 - x1)
        furniture_y2 = furniture_y1 + (y2 - y1)
        
        furniture_rgb_crop = furniture_rgb[furniture_y1:furniture_y2, furniture_x1:furniture_x2]
        furniture_alpha_crop = furniture_alpha[furniture_y1:furniture_y2, furniture_x1:furniture_x2]
        
        # Create composite with Alpha blending
        composite = background_np.copy()
        roi = composite[y1:y2, x1:x2]
        alpha = furniture_alpha_crop[:, :, np.newaxis] / 255.0
        composite[y1:y2, x1:x2] = (furniture_rgb_crop * alpha + roi * (1 - alpha)).astype(np.uint8)
        
        # Create mask (furniture area at new position)
        mask_img = np.zeros((background.height, background.width), dtype=np.uint8)
        mask_img[y1:y2, x1:x2] = furniture_alpha_crop
        
        # Save temporary files
        temp_composite_path = "output/temp_composite.png"
        temp_mask_path = "output/temp_placement_mask.png"
        os.makedirs("output", exist_ok=True)
        
        Image.fromarray(composite).save(temp_composite_path)
        Image.fromarray(mask_img).save(temp_mask_path)
        
        try:
            # Call Replicate API
            import replicate
            
            print("   ⏳ Calling Replicate API...")
            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                input={
                    "image": open(temp_composite_path, "rb"),
                    "mask": open(temp_mask_path, "rb"),
                    "prompt": "realistic furniture placement, natural lighting, proper perspective, matching shadows, interior scene",
                    "negative_prompt": "blurry, distorted, floating, unrealistic shadows, wrong perspective",
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5
                }
            )
            
            # Download result
            if isinstance(output, list) and len(output) > 0:
                output_url = output[0]
            else:
                output_url = output
            
            response = requests.get(output_url)
            result_img = Image.open(io.BytesIO(response.content))
            result_img.save(output_path)
            
            print(f"   ✅ AI blending complete")
            
            # Clean up temporary files
            os.remove(temp_composite_path)
            os.remove(temp_mask_path)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ AI blending failed: {e}")
            print(f"   → Downgrading to Poisson Blending")
            # Downgrade to Poisson approach
            return self.poisson_blend(furniture_img_path, background_img_path, 
                                     x, y, original_x, original_y, output_path)


if __name__ == "__main__":
    # Test code
    placer = FurniturePlacer()
    
    # Test Poisson Blending
    print("\n=== Test Poisson Blending ===")
    result = placer.poisson_blend(
        "output/furniture/furniture.png",
        "output/backgrounds/clean_bg.png",
        x=500, y=400,
        original_x=300, original_y=350,
        output_path="output/test_poisson.png"
    )
    print(f"Result saved to: {result}")
