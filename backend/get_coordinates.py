"""
图片坐标查看工具
用法：python get_coordinates.py image.png
"""
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_coordinates(image_path):
    """显示图片并实时显示鼠标坐标"""
    img = Image.open(image_path)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f'点击图片查看坐标\n图片尺寸: {img.width} x {img.height}', fontsize=14)
    
    # 显示坐标的文本
    coord_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                        fontsize=12)
    
    # 点击标记
    marker = None
    
    def on_move(event):
        """鼠标移动时更新坐标"""
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            coord_text.set_text(f'坐标: ({x}, {y})')
            fig.canvas.draw_idle()
    
    def on_click(event):
        """点击时标记位置并打印坐标"""
        nonlocal marker
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            print(f"\n✅ 点击坐标: ({x}, {y})")
            print(f"   测试命令: python test_sam.py {image_path} {x} {y}")
            
            # 移除旧标记
            if marker:
                marker.remove()
            
            # 添加新标记
            marker = ax.plot(x, y, 'r+', markersize=20, markeredgewidth=3)[0]
            coord_text.set_text(f'已选择: ({x}, {y})')
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("=" * 60)
    print(f"📷 图片: {image_path}")
    print(f"📐 尺寸: {img.width} x {img.height}")
    print("=" * 60)
    print("💡 移动鼠标查看实时坐标")
    print("💡 点击要分割的物体，会显示测试命令")
    print("💡 关闭窗口退出")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python get_coordinates.py <图片路径>")
        print("例如: python get_coordinates.py image.png")
        sys.exit(1)
    
    show_coordinates(sys.argv[1])
