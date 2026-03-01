"""
Cloud Inpainting Module - Using Replicate API + OpenCV Fallback
"""
import os
import requests
import numpy as np
from PIL import Image
import io
import cv2
from dotenv import load_dotenv

load_dotenv()

class CloudInpainter:
    def __init__(self):
        """Initialize cloud Inpainting"""
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        
        if self.replicate_token:
            print("✅ Cloud Inpainting initialized successfully (prioritizes Replicate)")
        else:
            print("⚠️  REPLICATE_API_TOKEN not found, will use OpenCV basic repair")
    
    def opencv_inpaint(self, image_np, mask_np):
        """
        Use OpenCV for basic inpainting (fast but moderate results)
        """
        print("⏳ Using OpenCV for basic repair...")
        result = cv2.inpaint(image_np, mask_np, 3, cv2.INPAINT_TELEA)
        return result
    
    def replicate_inpaint(self, image_path, mask_path):
        """
        Use Replicate API for inpainting
        """
        try:
            import replicate
            
            print("⏳ Calling Replicate API...")
            
            # Use model that supports inpainting
            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting",
                input={
                    "image": open(image_path, "rb"),
                    "mask": open(mask_path, "rb"),
                    "prompt": "empty clean room, interior, no furniture, smooth walls, clean floor",
                    "negative_prompt": "furniture, objects, desk, chair, table, cluttered",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5
                }
            )
            
            # Download result
            if output:
                response = requests.get(output)
                return Image.open(io.BytesIO(response.content))
            
        except Exception as e:
            print(f"⚠️  Replicate API failed: {str(e)}")
            return None
    
    def inpaint(self, image_path, mask_input, output_path):
        """
        Intelligent inpainting: prioritize Replicate, fallback to OpenCV
        
        Args:
            image_path: Original image path
            mask_input: SAM mask (numpy array, boolean) or mask image path (str)
            output_path: Output path
        
        Returns:
            str: Output file path
        """
        # 1. Read image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # 2. Handle mask input (supports array or path)
        if isinstance(mask_input, str):
            # If it's a path, read PNG's alpha channel as mask
            mask_img = Image.open(mask_input)
            if mask_img.mode == 'RGBA':
                mask_array = np.array(mask_img.split()[-1]) > 128
            else:
                mask_array = np.array(mask_img.convert('L')) > 128
            mask_uint8 = (mask_array * 255).astype(np.uint8)
        else:
            # If it's an array, convert directly
            mask_uint8 = (mask_input * 255).astype(np.uint8)
        
        # Ensure size consistency
        if mask_uint8.shape != image_np.shape[:2]:
            mask_uint8 = cv2.resize(mask_uint8, (image_np.shape[1], image_np.shape[0]))
        
        result_image = None
        
        # 3. Try Replicate API
        if self.replicate_token:
            # Save temporary mask file
            mask_image = Image.fromarray(mask_uint8, mode='L')
            temp_mask_path = output_path.replace(".png", "_mask_temp.png")
            mask_image.save(temp_mask_path)
            
            result_image = self.replicate_inpaint(image_path, temp_mask_path)
            
            # Clean up temporary file
            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)
        
        # 4. Fallback to OpenCV
        if result_image is None:
            print("⏳ Using OpenCV basic repair (Fallback)...")
            result_np = self.opencv_inpaint(image_np, mask_uint8)
            result_image = Image.fromarray(result_np)
        
        # 5. Save result
        result_image.save(output_path)
        print(f"✅ Inpainting complete! Saved to: {output_path}")
        return output_path
    
    def process_full_pipeline(self, image_path, mask_array, output_dir="output"):
        """
        Complete pipeline: Receive SAM mask → inpainting → Return result
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/masks", exist_ok=True)
        os.makedirs(f"{output_dir}/backgrounds", exist_ok=True)
        
        # Save mask
        mask_uint8 = (mask_array * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8)
        mask_path = f"{output_dir}/masks/mask_cloud.png"
        mask_image.save(mask_path)
        print(f"📦 Mask saved: {mask_path}")
        
        # Execute inpainting
        background_path = f"{output_dir}/backgrounds/clean_background_cloud.png"
        self.inpaint(image_path, mask_array, background_path)
        
        return {
            "background_clean": background_path,
            "mask": mask_path
        }


# Convenience function
def remove_object_cloud(image_path, mask_array, output_dir="output"):
    """
    Cloud Inpainting complete in one step
    """
    inpainter = CloudInpainter()
    return inpainter.process_full_pipeline(image_path, mask_array, output_dir)
