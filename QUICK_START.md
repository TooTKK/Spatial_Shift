# 🚀 Spatial Shift - 快速启动指南

## 📋 项目功能
AI 智能家具移动工具 - 点击选中家具，再点击目标位置，AI 自动完成移动并生成逼真效果图

## ⚙️ 环境要求
- **Python**: 3.11+
- **Node.js**: 16+ 
- **Replicate API Token**: 用于 Inpainting（可选）

---

## 🎯 启动步骤

### 1️⃣ 启动后端 API

```bash
# 进入后端目录
cd backend

# 确保已安装依赖（首次运行）
pip install -r ../requirements.txt

# 启动 FastAPI 服务器
python3 main.py
```

**后端将运行在:** `http://localhost:8000`

**检查状态:** 浏览器打开 `http://localhost:8000` 应该看到:
```json
{
  "service": "Spatial Shift API",
  "status": "running",
  "endpoints": [...]
}
```

---

### 2️⃣ 启动前端界面

**打开新终端窗口：**

```bash
# 进入前端目录
cd frontend

# 安装依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

**前端将运行在:** `http://localhost:5173`（或其他端口，终端会显示）

**浏览器打开前端地址即可使用！**

---

## 🎨 使用流程

### 步骤 1: 上传图片
- 点击 "Choose picture" 上传室内照片
- 点击 "Confirm" 确认

### 步骤 2: 第一次点击（选择家具）
- 鼠标点击你想移动的家具
- 会出现 **红色圆圈标记 "1"**
- 提示: "🖱️ Step 1: Click on the furniture you want to move"

### 步骤 3: 第二次点击（选择目标位置）
- 再点击你想放置的位置
- 会出现 **蓝色圆圈标记 "2"**
- 提示: "🎯 Step 2: Click where you want to place it"

### 步骤 4: 自动处理
后端会自动：
1. 🔍 SAM 智能分割家具（包括椅子腿、靠枕等相关物体）
2. 🎨 Inpainting 移除家具并修复背景
3. ✨ Alpha Blending 将家具放置到新位置

### 步骤 5: 查看结果
- 显示最终效果图
- 点击 "Move another furniture" 可以用同一张图移动其他家具
- 点击 "×" 关闭重新开始

---

## 🔧 后端 API 端点

### `POST /upload`
上传室内照片
- **输入**: `file` (multipart/form-data)
- **输出**: `{ "image_id": "xxx", "filename": "..." }`

### `POST /full_pipeline`
完整流程（推荐）
- **输入**:
  - `file`: 图片文件
  - `segment_x`, `segment_y`: 家具位置坐标
  - `place_x`, `place_y`: 目标位置坐标
  - `use_ai`: `false`（使用 Alpha Blending，推荐）
- **输出**:
  - `final_image`: Base64 编码的最终图片
  - `furniture_mask`: Base64 编码的家具抠图
  - `clean_background`: Base64 编码  的干净背景
  - `method`: 使用的方法

### 其他端点
- `POST /segment`: 单独分割家具
- `POST /remove_furniture`: 单独移除家具
- `POST /place_furniture`: 单独放置家具

---

## 🐛 常见问题

### 后端启动失败
```bash
# 检查 Python 版本
python3 --version  # 应 >= 3.11

# 重新安装依赖
pip install --upgrade -r requirements.txt

# 检查 SAM 模型是否存在
ls backend/checkpoints/sam2.1_hiera_large.pt
```

### 前端无法连接后端
1. 确认后端在运行（`http://localhost:8000` 能访问）
2. 检查前端 API 地址配置（`src/App.vue` 第 5 行）
3. 浏览器控制台查看 CORS 错误

### Inpainting 失败
- Inpainting 使用 Replicate Cloud API（需要 API Token）
- 设置环境变量: `export REPLICATE_API_TOKEN=your_token`
- 如果没有 token，会自动降级到 OpenCV inpainting

### 图片坐标不准确
- 前端已自动处理缩放问题（`actualX = displayX * scaleX`）
- 如果仍然不准，检查图片是否被 CSS 变形

---

## 📁 项目结构

```
Spatial_Shift/
├── backend/
│   ├── main.py                 # FastAPI 服务器
│   ├── sam.py                  # SAM2 分割（智能识别相关物体）
│   ├── inpainting_cloud.py     # Replicate/OpenCV 背景修复
│   ├── furniture_placement.py  # Alpha Blending 放置
│   ├── checkpoints/            # SAM2 模型文件
│   ├── output/                 # 输出文件夹
│   └── uploads/                # 上传的图片
├── frontend/
│   ├── src/
│   │   └── App.vue            # 主界面（两次点击逻辑）
│   ├── package.json
│   └── vite.config.js
└── QUICK_START.md             # 本文档
```

---

## 🎓 技术栈

**后端:**
- FastAPI - REST API 框架
- SAM 2.1 - Meta 的 Segment Anything 模型
- Replicate API - Cloud Inpainting (flux-fill-pro)
- OpenCV - Alpha Blending 和降级 Inpainting
- PIL/Pillow - 图像处理

**前端:**
- Vue 3 - UI 框架
- Vite - 开发服务器

---

## 💡 开发建议

- 使用 `use_ai=false` 参数（Alpha Blending 快速稳定）
- SAM 智能分割会自动识别相关物体（椅子腿、靠枕等）
- 深度缩放：家具会根据 Y 坐标自动调整大小（模拟透视）
- 坐标系统：左上角为 (0, 0)，向右向下递增

---

## 📞 技术支持

如有问题，检查：
1. 浏览器控制台（前端错误）
2. 后端终端输出（API 错误）
3. `backend/output/` 目录（中间结果图片）

祝使用愉快！🎉
