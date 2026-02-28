"""
云端 Inpainting 模块 - 使用 Replicate API + OpenCV Fallback
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
        """初始化云端 Inpainting"""
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        
        if self.replicate_token:
            print("✅ 云端 Inpainting 初始化成功（优先使用 Replicate）")
        else:
            print("⚠️  未找到 REPLICATE_API_TOKEN，将使用 OpenCV 基础修复")
    
    def opencv_inpaint(self, image_np, mask_np):
        """
        使用 OpenCV 进行基础 inpainting（快速但效果一般）
        """
        print("⏳ 使用 OpenCV 进行基础修复...")
        result = cv2.inpaint(image_np, mask_np, 3, cv2.INPAINT_TELEA)
        return result
    
    def replicate_inpaint(self, image_path, mask_path):
        """
        使用 Replicate API 进行 inpainting
        """
        try:
            import replicate
            
            print("⏳ 调用 Replicate API...")
            
            # 使用支持 inpainting 的模型
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
            
            # 下载结果
            if output:
                response = requests.get(output)
                return Image.open(io.BytesIO(response.content))
            
        except Exception as e:
            print(f"⚠️  Replicate API 失败: {str(e)}")
            return None
    
    def inpaint(self, image_path, mask_array, output_path):
        """
        智能 inpainting：优先 Replicate，fallback 到 OpenCV
        
        Args:
            image_path: 原始图片路径
            mask_array: SAM 输出的 mask (numpy array, boolean)
            output_path: 输出路径
        
        Returns:
            str: 输出文件路径
        """
        # 1. 读取图片
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # 2. 创建 mask
        mask_uint8 = (mask_array * 255).astype(np.uint8)
        
        # 确保尺寸一致
        if mask_uint8.shape != image_np.shape[:2]:
            mask_uint8 = cv2.resize(mask_uint8, (image_np.shape[1], image_np.shape[0]))
        
        result_image = None
        
        # 3. 尝试 Replicate API
        if self.replicate_token:
            # 保存临时 mask 文件
            mask_image = Image.fromarray(mask_uint8, mode='L')
            temp_mask_path = output_path.replace(".png", "_mask_temp.png")
            mask_image.save(temp_mask_path)
            
            result_image = self.replicate_inpaint(image_path, temp_mask_path)
            
            # 清理临时文件
            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)
        
        # 4. Fallback 到 OpenCV
        if result_image is None:
            print("⏳ 使用 OpenCV 基础修复（Fallback）...")
            result_np = self.opencv_inpaint(image_np, mask_uint8)
            result_image = Image.fromarray(result_np)
        
        # 5. 保存结果
        result_image.save(output_path)
        print(f"✅ Inpainting 完成！保存到: {output_path}")
        return output_path
    
    def process_full_pipeline(self, image_path, mask_array, output_dir="output"):
        """
        完整流程：接收 SAM mask → inpainting → 返回结果
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/masks", exist_ok=True)
        os.makedirs(f"{output_dir}/backgrounds", exist_ok=True)
        
        # 保存 mask
        mask_uint8 = (mask_array * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8)
        mask_path = f"{output_dir}/masks/mask_cloud.png"
        mask_image.save(mask_path)
        print(f"📦 Mask 已保存: {mask_path}")
        
        # 执行 inpainting
        background_path = f"{output_dir}/backgrounds/clean_background_cloud.png"
        self.inpaint(image_path, mask_array, background_path)
        
        return {
            "background_clean": background_path,
            "mask": mask_path
        }


# 便捷函数
def remove_object_cloud(image_path, mask_array, output_dir="output"):
    """
    云端 Inpainting 一步完成
    """
    inpainter = CloudInpainter()
    return inpainter.process_full_pipeline(image_path, mask_array, output_dir)
