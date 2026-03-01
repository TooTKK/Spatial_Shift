# Spatial Shift 🪑✨

**AI室内家具智能移动可视化工具**

上传室内照片 → 点击选择家具 → AI分割 → 拖拽到新位置 → AI智能融合生成效果图

---

## 🎯 项目功能

- ✅ **智能分割**：SAM 2.1 Hiera Large精准识别家具（M4 Mac MPS加速）
- ✅ **背景修复**：Replicate Stable Diffusion Inpainting + OpenCV降级方案
- ✅ **智能放置**：
  - 🚀 **Poisson Blending**（快速，~1秒）
  - 🤖 **AI融合**（高质量，自动处理透视/光照/阴影，~30秒）
- ✅ **完整API**：FastAPI RESTful接口，支持Base64返回
- ✅ **自动缩放**：根据Y坐标深度估算智能调整家具比例

---

## 🏗️ 技术栈

| 模块 | 技术 |
|------|------|
| **后端框架** | Python 3.11 + FastAPI + uvicorn |
| **分割模型** | SAM 2.1 Hiera Large（本地856MB） |
| **背景修复** | Replicate API (SD Inpainting) + OpenCV |
| **家具放置** | Poisson Blending + SD Inpainting |
| **前端** | Vue 3 + Konva.js（队友负责） |
| **硬件加速** | M4 Mac MPS |

---

## 📦 安装

### 1. 克隆项目

```bash
git clone https://github.com/TooTKK/Spatial_Shift.git
cd Spatial_Shift
```

### 2. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 3. 安装 SAM2

```bash
pip install -e ./sam2
```

参考官方文档：https://github.com/facebookresearch/sam2

### 4. 下载模型

```bash
cd checkpoints
# SAM 2.1 Hiera Large 模型（856MB）
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

或者手动下载后放到 `backend/checkpoints/` 目录

### 5. 配置环境变量

创建 `backend/.env` 文件：

```bash
REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxx  # 从 replicate.com 获取（可选）
```

没有API token也能运行，会降级到OpenCV基础方案。

---

## 🚀 启动服务

### 后端

```bash
cd backend
python main.py
```

服务运行在 `http://localhost:8000`

查看API文档：http://localhost:8000/docs

### 前端

```bash
cd frontend
npm install
npm run dev
```

---

## 📚 使用指南

### 方式1：API调用（推荐）

详细文档：[backend/API_DOC.md](backend/API_DOC.md)

**快速示例：**

```bash
# 1. 上传图片
curl -X POST "http://localhost:8000/upload" -F "file=@room.jpg"
# 返回: {"image_id": "xxx"}

# 2. 分割家具（点击坐标 300, 350）
curl -X POST "http://localhost:8000/segment" \
  -F "image_id=xxx" -F "x=300" -F "y=350"
# 返回: {"furniture_mask": "base64...", "bbox": [x1,y1,x2,y2]}

# 3. 移除家具
curl -X POST "http://localhost:8000/remove_furniture" \
  -F "image_id=xxx" -F "furniture_mask_path=output/furniture/xxx.png"
# 返回: {"clean_background": "base64..."}

# 4. 放置到新位置（600, 400）
curl -X POST "http://localhost:8000/place_furniture" \
  -F "image_id=xxx" \
  -F "furniture_mask_path=output/furniture/xxx.png" \
  -F "background_path=output/backgrounds/xxx.png" \
  -F "new_x=600" -F "new_y=400" \
  -F "original_x=300" -F "original_y=350" \
  -F "use_ai=true"
# 返回: {"final_image": "base64...", "method": "ai_blend"}
```

### 方式2：命令行测试

```bash
cd backend
python test_placement.py
```

---

## 🎨 工作流程

```
用户上传照片
    ↓
点击家具 (x, y)
    ↓
SAM2分割 → 透明PNG家具抠图
    ↓
前端显示红色边界框
    ↓
调用Inpainting → 移除原位置家具 → 干净背景
    ↓
点击新位置 (new_x, new_y)
    ↓
智能放置（自动缩放 + Poisson/AI融合）
    ↓
返回最终效果图
```

---

## 📂 项目结构

```
Spatial_Shift/
├── backend/
│   ├── main.py                    # FastAPI主服务
│   ├── sam.py                     # SAM2分割模块
│   ├── inpainting_cloud.py        # 背景修复模块
│   ├── furniture_placement.py     # 家具放置模块（新增）
│   ├── test_placement.py          # 完整流程测试（新增）
│   ├── API_DOC.md                 # API详细文档（新增）
│   ├── checkpoints/
│   │   └── sam2.1_hiera_large.pt  # SAM2模型
│   ├── output/
│   │   ├── furniture/             # 家具抠图
│   │   ├── backgrounds/           # 干净背景
│   │   └── placed/                # 最终效果图
│   └── uploads/                   # 用户上传的图片
├── frontend/                      # Vue3前端（队友开发）
├── sam2/                          # SAM2源码
└── requirements.txt               # Python依赖
```

---

## 🔧 家具放置方案对比

| 方案 | 速度 | 效果 | 成本 | 适用场景 |
|------|------|------|------|----------|
| **Poisson Blending** | ⚡ ~1秒 | ⭐⭐⭐ | 免费 | 实时预览、快速原型 |
| **AI Blend (SD)** | 🐌 ~30秒 | ⭐⭐⭐⭐⭐ | $0.01/次 | 最终输出、展示 |

**推荐策略：** 开发时用Poisson快速测试，最终生成时切换到AI融合

---

## ⚙️ 核心算法

### 自动深度缩放

根据Y坐标估算物体深度，自动调整家具大小：

```python
scale = 0.5 + (y / image_height) * 1.0

# 示例：
# 从上方（Y=200）移到下方（Y=800），图片高1000px
# 原缩放 = 0.5 + 0.2 = 0.7
# 新缩放 = 0.5 + 0.8 = 1.3
# 相对缩放 = 1.3 / 0.7 ≈ 1.86倍（家具变大）
```

---

## 🐛 常见问题

### Q: API返回500错误？

检查：
1. SAM2模型是否下载到 `backend/checkpoints/`
2. SAM2是否正确安装：`pip install -e ./sam2`
3. 查看终端日志定位错误

### Q: AI融合失败？

- 检查 `.env` 中的 `REPLICATE_API_TOKEN` 是否正确
- 没有token会自动降级到Poisson Blending
- 查看终端输出的错误信息

### Q: 家具放置后比例不对？

确保传递了 `original_x, original_y` 参数，系统依赖这些值估算缩放

---

## 🚀 下一步计划

- [ ] 前后端完整对接
- [ ] 支持多个家具同时移动
- [ ] 手动调整缩放/旋转
- [ ] 使用MiDaS深度估计改进缩放算法
- [ ] 添加阴影生成
- [ ] WebSocket实时推送进度

---

## 📄 许可证

MIT License

---

## 👥 团队

- **后端开发**：你（SAM2分割 + Inpainting + 家具放置）
- **前端开发**：队友（Vue3 + Konva.js）

---

## 🔗 相关链接

- [FastAPI自动文档](http://localhost:8000/docs)
- [SAM2官方仓库](https://github.com/facebookresearch/sam2)
- [Replicate API](https://replicate.com/)

---

**享受AI的魔法吧！✨🪄**