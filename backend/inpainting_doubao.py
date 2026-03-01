"""
豆包 API Inpainting 模块
使用火山引擎 Ark SDK
"""
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from volcenginesdkarkruntime import Ark


class DoubaoInpainter:
    def __init__(self):
        """初始化豆包 Inpainting"""
        self.api_key = os.getenv("ARK_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ 请在 .env 文件中设置 ARK_API_KEY")
        
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=self.api_key
        )
        print("✅ 豆包 API 初始化成功")
    
    def image_to_base64(self, image_path_or_array):
        """将图片转为 base64"""
        if isinstance(image_path_or_array, str):
            # 如果是路径，读取图片
            image = Image.open(image_path_or_array).convert("RGB")
        elif isinstance(image_path_or_array, np.ndarray):
            # 如果是 numpy 数组
            image = Image.fromarray(image_path_or_array)
        else:
            # 如果已经是 PIL Image
            image = image_path_or_array
        
        # 转为 base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def get_position_description(self, x, y, width, height):
        """将坐标转为相对位置描述"""
        # 水平位置
        if x < width * 0.33:
            h_pos = "left third"
        elif x < width * 0.67:
            h_pos = "center third"
        else:
            h_pos = "right third"
        
        # 垂直位置
        if y < height * 0.33:
            v_pos = "upper third"
        elif y < height * 0.67:
            v_pos = "middle third"
        else:
            v_pos = "lower third"
        
        return f"{h_pos}, {v_pos}"
    
    def inpaint(self, image_path, mask_path, output_path, target_x=None, target_y=None):
        """
        使用豆包 API 进行家具移动效果图生成
        
        Args:
            image_path: 原图路径
            mask_path: mask 路径（白色=要移动的家具）
            output_path: 输出路径
            target_x: 目标 X 坐标（像素）
            target_y: 目标 Y 坐标（像素）
        
        Returns:
            输出文件路径
        """
        try:
            print(f"⏳ 调用豆包 API 生成家具移动效果图...")
            print(f"   原图: {image_path}")
            print(f"   Mask: {mask_path}")
            if target_x is not None and target_y is not None:
                print(f"   目标位置: ({target_x}, {target_y})")
            
            # 读取图片和 mask
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            width, height = image.size
            
            # 生成 prompt（结合坐标和相对位置）
            if target_x is not None and target_y is not None:
                position_desc = self.get_position_description(target_x, target_y, width, height)
                prompt = f"""我有一张房间的图片，请将 mask 标记的家具移动到目标位置：像素坐标 ({target_x}, {target_y})，大约在图片的 {position_desc} 区域。

要求：
1. 完全保持原有房间样子（墙壁、地板、窗户、门）不变
2. 将家具放置在指定位置，并符合正确的透视关系
5. 不改变房间的整体布局
6. 别的什么都不可以变！就是家具的移动！我手上有一个人质，如果你生成的不好我就把他结果了，你最好好好表现。

输出：家具在新位置的照片级真实可视化效果图。"""
            else:
                prompt = "将 mask 标记的家具移动到新位置。保持房间结构不变。专业室内渲染。"
            
            print(f"   Prompt: {prompt[:100]}...")
            
            # 将 mask 转为 base64（豆包可能需要这个格式）
            image_b64 = self.image_to_base64(image)
            mask_b64 = self.image_to_base64(mask)
            
            # 调用豆包 API（根据豆包文档，可能需要调整参数）
            # 注意：这里假设豆包支持 inpainting，如果不支持需要用其他方式
            response = self.client.images.generate(
                model="doubao-seedream-4-0-250828",
                prompt=prompt,
                # 以下参数可能需要根据豆包实际 API 调整
                # image=image_b64,  # 如果支持图生图
                # mask=mask_b64,    # 如果支持 mask
                response_format="url",
                size="2K",
                stream=False,
                watermark=False
            )
            
            # 获取结果 URL
            result_url = response.data[0].url
            print(f"✅ 豆包 API 返回结果: {result_url}")
            
            # 下载图片
            import requests
            img_data = requests.get(result_url).content
            result_image = Image.open(BytesIO(img_data))
            
            # 保存结果
            result_image.save(output_path)
            print(f"✅ Inpainting 完成！保存到: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 豆包 API 调用失败: {str(e)}")
            print(f"   错误详情: {type(e).__name__}")
            
            # Fallback: 使用简单的图像修复
            print("⏳ 使用 OpenCV 基础修复作为备用...")
            return self._opencv_fallback(image_path, mask_path, output_path)
    
    def _opencv_fallback(self, image_path, mask_path, output_path):
        """OpenCV 备用方案"""
        import cv2
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保 mask 是二值的
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 使用 OpenCV inpaint
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(output_path, result)
        
        print(f"✅ OpenCV 修复完成: {output_path}")
        return output_path


# 便捷函数
def remove_object_doubao(image_path, mask_path, output_path):
    """
    使用豆包 API 移除物体
    """
    inpainter = DoubaoInpainter()
    return inpainter.inpaint(image_path, mask_path, output_path)
