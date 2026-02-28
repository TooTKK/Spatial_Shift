"""
SAM 2 本地测试脚本
用法：python test_sam.py <图片路径> <x坐标> <y坐标>
例如：python test_sam.py test_image.jpg 500 300
"""
import sys
import os

# 添加 sam2 到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "../sam2"))

from sam import SAM2Handler

def test_segmentation(image_path, x, y):
    """测试 SAM 分割功能"""
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return
    
    print("=" * 60)
    print("🚀 开始测试 SAM 2 分割功能")
    print("=" * 60)
    print(f"📷 输入图片: {image_path}")
    print(f"📍 点击坐标: ({x}, {y})")
    print()
    
    # 配置路径
    CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    try:
        # 初始化 SAM 处理器
        print("⏳ 正在加载 SAM 2 模型...")
        sam_engine = SAM2Handler(CHECKPOINT, MODEL_CONFIG)
        print("✅ 模型加载成功！")
        print()
        
        # 创建输出目录
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", f"furniture_{os.path.basename(image_path).rsplit('.', 1)[0]}.png")
        
        # 执行分割
        print("⏳ 正在执行分割...")
        result_path, score = sam_engine.process_segmentation(image_path, x, y, output_path)
        
        print("=" * 60)
        print("✅ 分割完成！")
        print("=" * 60)
        print(f"📁 输出文件: {result_path}")
        print(f"🎯 置信度分数: {score:.4f}")
        print()
        print("💡 提示: 打开输出文件查看分割出的家具（透明背景PNG）")
        
    except Exception as e:
        print("=" * 60)
        print("❌ 测试失败！")
        print("=" * 60)
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("使用方法: python test_sam.py <图片路径> <x坐标> <y坐标>")
        print("例如: python test_sam.py test_image.jpg 500 300")
        print()
        print("提示: 坐标是你想要分割的物体上的任意一点")
        sys.exit(1)
    
    image_path = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    
    test_segmentation(image_path, x, y)
