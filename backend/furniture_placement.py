"""
家具智能放置模块
支持两种方案：
1. Poisson Blending（快速原型）
2. Stable Diffusion（智能融合）
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
        """初始化家具放置器"""
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if self.replicate_token:
            print("✅ 家具放置器初始化成功（支持AI融合）")
        else:
            print("⚠️  未找到 REPLICATE_API_TOKEN，仅支持基础融合")
    
    def estimate_scale(self, original_y, new_y, image_height):
        """
        根据Y坐标估算深度缩放
        原理：室内场景中，Y越大（越下方）通常表示越近，物体应该越大
        """
        # 简化版透视缩放：基于Y坐标的线性插值
        # 假设图像底部是前景（scale=1.2），顶部是远景（scale=0.5）
        original_ratio = original_y / image_height
        new_ratio = new_y / image_height
        
        # 基础缩放因子（0.5到1.5之间）
        original_scale = 0.5 + original_ratio * 1.0
        new_scale = 0.5 + new_ratio * 1.0
        
        # 相对缩放
        relative_scale = new_scale / original_scale
        return np.clip(relative_scale, 0.3, 3.0)  # 限制缩放范围
    
    def poisson_blend(self, furniture_img_path, background_img_path, x, y, 
                      original_x=None, original_y=None, output_path="output/placed.png"):
        """
        简化版：直接使用Alpha混合进行家具放置（最稳定）
        
        Args:
            furniture_img_path: 透明PNG家具抠图路径（原图大小，椅子在原位置，其他透明）
            background_img_path: 干净背景图路径
            x, y: 新位置中心坐标
            original_x, original_y: 原位置坐标（用于估算缩放）
            output_path: 输出路径
        
        Returns:
            输出图片路径
        """
        print("⏳ 使用 Alpha Blending 进行家具放置...")
        
        # 读取图片
        furniture_full = Image.open(furniture_img_path).convert("RGBA")
        background = Image.open(background_img_path).convert("RGB")
        
        # 1. 先从原图大小的PNG中裁剪出家具的实际边界框
        furniture_np = np.array(furniture_full)
        alpha_channel = furniture_np[:, :, 3]
        
        # 找到非透明区域的边界
        rows = np.any(alpha_channel > 0, axis=1)
        cols = np.any(alpha_channel > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            print("   ⚠️  未找到非透明区域")
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # 裁剪出家具部分
        furniture = furniture_full.crop((x_min, y_min, x_max + 1, y_max + 1))
        print(f"   ✂️  从原图裁剪家具: 原位置({x_min}, {y_min}), 尺寸{furniture.width}x{furniture.height}")
        
        # 自动缩放：根据Y坐标估算深度
        if original_x and original_y:
            scale = self.estimate_scale(original_y, y, background.height)
            new_width = int(furniture.width * scale)
            new_height = int(furniture.height * scale)
            furniture = furniture.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"   📏 自动缩放: {scale:.2f}x (原Y={original_y}, 新Y={y})")
        
        # 转换为numpy数组
        furniture_np = np.array(furniture)
        background_np = np.array(background)
        
        # 提取RGB和Alpha通道
        furniture_rgb = furniture_np[:, :, :3]
        furniture_alpha = furniture_np[:, :, 3:4] / 255.0  # 归一化到 0-1
        
        # 计算放置位置（中心对齐）
        center_x = int(x)
        center_y = int(y)
        half_w = furniture.width // 2
        half_h = furniture.height // 2
        
        # 计算目标区域范围
        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = x1 + furniture.width
        y2 = y1 + furniture.height
        
        # 处理边界情况：计算有效范围
        bg_h, bg_w = background_np.shape[:2]
        
        # 背景上的有效区域
        bg_x1 = max(0, x1)
        bg_y1 = max(0, y1)
        bg_x2 = min(bg_w, x2)
        bg_y2 = min(bg_h, y2)
        
        # 家具上对应的有效区域
        furn_x1 = bg_x1 - x1
        furn_y1 = bg_y1 - y1
        furn_x2 = furn_x1 + (bg_x2 - bg_x1)
        furn_y2 = furn_y1 + (bg_y2 - bg_y1)
        
        # 检查是否有有效区域
        if bg_x2 <= bg_x1 or bg_y2 <= bg_y1:
            print("   ⚠️  家具完全超出背景边界")
            result = Image.fromarray(background_np)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.save(output_path)
            return output_path
        
        # 提取对应区域
        bg_roi = background_np[bg_y1:bg_y2, bg_x1:bg_x2]
        furn_rgb_roi = furniture_rgb[furn_y1:furn_y2, furn_x1:furn_x2]
        furn_alpha_roi = furniture_alpha[furn_y1:furn_y2, furn_x1:furn_x2]
        
        # Alpha混合：保留所有细节（包括半透明椅子腿）
        # result = foreground * alpha + background * (1 - alpha)
        blended_roi = (furn_rgb_roi * furn_alpha_roi + 
                       bg_roi * (1 - furn_alpha_roi)).astype(np.uint8)
        
        # 将混合结果放回背景
        result_np = background_np.copy()
        result_np[bg_y1:bg_y2, bg_x1:bg_x2] = blended_roi
        
        # 保存结果
        result = Image.fromarray(result_np)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path)
        
        print(f"   ✅ 家具放置成功: {output_path}")
        return output_path
    
    def ai_blend(self, furniture_img_path, background_img_path, x, y, 
                 original_x=None, original_y=None, output_path="output/placed_ai.png"):
        """
        方案A：使用Stable Diffusion进行智能融合
        
        原理：在新位置创建mask，用家具图+背景让AI重绘，自动处理透视、光照
        
        Args:
            furniture_img_path: 透明PNG家具抠图路径
            background_img_path: 干净背景图路径
            x, y: 新位置中心坐标
            original_x, original_y: 原位置坐标（用于估算缩放）
            output_path: 输出路径
        
        Returns:
            输出图片路径
        """
        if not self.replicate_token:
            print("❌ 需要 REPLICATE_API_TOKEN 才能使用AI融合")
            return None
        
        print("⏳ 使用 Stable Diffusion 进行智能家具放置...")
        
        # 读取图片
        furniture = Image.open(furniture_img_path).convert("RGBA")
        background = Image.open(background_img_path).convert("RGB")
        
        # 估算缩放（与Poisson方案一致）
        if original_x and original_y:
            scale = self.estimate_scale(original_y, y, background.height)
            new_width = int(furniture.width * scale)
            new_height = int(furniture.height * scale)
            furniture = furniture.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"   📏 预缩放: {scale:.2f}x")
        
        # 先用Alpha混合创建初步合成（作为SD的输入图）
        furniture_rgb = np.array(furniture.convert("RGB"))
        furniture_alpha = np.array(furniture.split()[-1])
        background_np = np.array(background)
        
        # 计算放置位置
        half_w = furniture.width // 2
        half_h = furniture.height // 2
        x1 = max(0, x - half_w)
        y1 = max(0, y - half_h)
        x2 = min(background.width, x + half_w)
        y2 = min(background.height, y + half_h)
        
        # 裁剪家具
        furniture_x1 = half_w - (x - x1)
        furniture_y1 = half_h - (y - y1)
        furniture_x2 = furniture_x1 + (x2 - x1)
        furniture_y2 = furniture_y1 + (y2 - y1)
        
        furniture_rgb_crop = furniture_rgb[furniture_y1:furniture_y2, furniture_x1:furniture_x2]
        furniture_alpha_crop = furniture_alpha[furniture_y1:furniture_y2, furniture_x1:furniture_x2]
        
        # Alpha混合创建合成图
        composite = background_np.copy()
        roi = composite[y1:y2, x1:x2]
        alpha = furniture_alpha_crop[:, :, np.newaxis] / 255.0
        composite[y1:y2, x1:x2] = (furniture_rgb_crop * alpha + roi * (1 - alpha)).astype(np.uint8)
        
        # 创建mask（新位置的家具区域）
        mask_img = np.zeros((background.height, background.width), dtype=np.uint8)
        mask_img[y1:y2, x1:x2] = furniture_alpha_crop
        
        # 保存临时文件
        temp_composite_path = "output/temp_composite.png"
        temp_mask_path = "output/temp_placement_mask.png"
        os.makedirs("output", exist_ok=True)
        
        Image.fromarray(composite).save(temp_composite_path)
        Image.fromarray(mask_img).save(temp_mask_path)
        
        try:
            # 调用Replicate API
            import replicate
            
            print("   ⏳ 调用 Replicate API...")
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
            
            # 下载结果
            if isinstance(output, list) and len(output) > 0:
                output_url = output[0]
            else:
                output_url = output
            
            response = requests.get(output_url)
            result_img = Image.open(io.BytesIO(response.content))
            result_img.save(output_path)
            
            print(f"   ✅ AI融合完成")
            
            # 清理临时文件
            os.remove(temp_composite_path)
            os.remove(temp_mask_path)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ AI融合失败: {e}")
            print(f"   → 降级使用Poisson Blending")
            # 降级到Poisson方案
            return self.poisson_blend(furniture_img_path, background_img_path, 
                                     x, y, original_x, original_y, output_path)


if __name__ == "__main__":
    # 测试代码
    placer = FurniturePlacer()
    
    # 测试Poisson Blending
    print("\n=== 测试 Poisson Blending ===")
    result = placer.poisson_blend(
        "output/furniture/furniture.png",
        "output/backgrounds/clean_bg.png",
        x=500, y=400,
        original_x=300, original_y=350,
        output_path="output/test_poisson.png"
    )
    print(f"结果保存至: {result}")
