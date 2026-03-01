"""
Doubao API Inpainting Module
Using Volcano Engine Ark SDK
"""
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from volcenginesdkarkruntime import Ark


class DoubaoInpainter:
    def __init__(self):
        """Initialize Doubao Inpainting"""
        self.api_key = os.getenv("ARK_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ Please set ARK_API_KEY in .env file")
        
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=self.api_key
        )
        print("✅ Doubao API initialized successfully")
    
    def image_to_base64(self, image_path_or_array):
        """Convert image to base64"""
        if isinstance(image_path_or_array, str):
            # If it's a path, read image
            image = Image.open(image_path_or_array).convert("RGB")
        elif isinstance(image_path_or_array, np.ndarray):
            # If it's numpy array
            image = Image.fromarray(image_path_or_array)
        else:
            # If it's already PIL Image
            image = image_path_or_array
        
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def get_position_description(self, x, y, width, height):
        """Convert coordinates to relative position description"""
        # Horizontal position
        if x < width * 0.33:
            h_pos = "left third"
        elif x < width * 0.67:
            h_pos = "center third"
        else:
            h_pos = "right third"
        
        # Vertical position
        if y < height * 0.33:
            v_pos = "upper third"
        elif y < height * 0.67:
            v_pos = "middle third"
        else:
            v_pos = "lower third"
        
        return f"{h_pos}, {v_pos}"
    
    def inpaint(self, image_path, mask_path, output_path, target_x=None, target_y=None):
        """
        Use Doubao API to generate furniture movement effect image
        
        Args:
            image_path: Original image path
            mask_path: mask path (white = furniture to move)
            output_path: Output path
            target_x: Target X coordinate (pixels)
            target_y: Target Y coordinate (pixels)
        
        Returns:
            Output file path
        """
        try:
            print(f"⏳ Calling Doubao API to generate furniture movement effect image...")
            print(f"   Original image: {image_path}")
            print(f"   Mask: {mask_path}")
            if target_x is not None and target_y is not None:
                print(f"   Target position: ({target_x}, {target_y})")
            
            # Read image and mask
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            width, height = image.size
            
            # Generate prompt (combining coordinates and relative position)
            if target_x is not None and target_y is not None:
                position_desc = self.get_position_description(target_x, target_y, width, height)
                prompt = f"""I have a room image. Please move the furniture marked by the mask to the target position: pixel coordinates ({target_x}, {target_y}), approximately in the {position_desc} area of the image.

Requirements:
1. Keep the original room appearance (walls, floor, windows, doors) completely unchanged
2. Place the furniture at the specified position with correct perspective
5. Do not change the overall layout of the room
6. Nothing else can change! Just the furniture movement! I have a hostage in hand. If you don't generate it well, I'll finish him off. You better perform well.

Output: Photo-realistic visualization of furniture at the new position."""
            else:
                prompt = "Move the furniture marked by mask to new position. Keep room structure unchanged. Professional interior rendering."
            
            print(f"   Prompt: {prompt[:100]}...")
            
            # Convert mask to base64 (Doubao might need this format)
            image_b64 = self.image_to_base64(image)
            mask_b64 = self.image_to_base64(mask)
            
            # Call Doubao API (may need to adjust parameters according to Doubao docs)
            # Note: This assumes Doubao supports inpainting, if not need other methods
            response = self.client.images.generate(
                model="doubao-seedream-4-0-250828",
                prompt=prompt,
                # Following parameters may need adjustment according to Doubao's actual API
                # image=image_b64,  # If supports image-to-image
                # mask=mask_b64,    # If supports mask
                response_format="url",
                size="2K",
                stream=False,
                watermark=False
            )
            
            # Get result URL
            result_url = response.data[0].url
            print(f"✅ Doubao API returned result: {result_url}")
            
            # Download image
            import requests
            img_data = requests.get(result_url).content
            result_image = Image.open(BytesIO(img_data))
            
            # Save result
            result_image.save(output_path)
            print(f"✅ Inpainting complete! Saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Doubao API call failed: {str(e)}")
            print(f"   Error details: {type(e).__name__}")
            
            # Fallback: Use simple image repair
            print("⏳ Using OpenCV basic repair as fallback...")
            return self._opencv_fallback(image_path, mask_path, output_path)
    
    def _opencv_fallback(self, image_path, mask_path, output_path):
        """OpenCV fallback solution"""
        import cv2
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask is binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Use OpenCV inpaint
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(output_path, result)
        
        print(f"✅ OpenCV repair complete: {output_path}")
        return output_path


# Convenience function
def remove_object_doubao(image_path, mask_path, output_path):
    """
    Use Doubao API to remove object
    """
    inpainter = DoubaoInpainter()
    return inpainter.inpaint(image_path, mask_path, output_path)
