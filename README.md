# Spatial Shift

An AI-powered interior furniture visualization tool that allows users to move furniture in room images and generate realistic renderings.

---

## Features

- **Smart Segmentation**: SAM 2.1 Hiera Large for accurate furniture detection
- **Background Inpainting**: Replicate API for background restoration
- **Furniture Placement**: Alpha Blending for fast compositing
- **Full Pipeline**: Upload → Click furniture → Click target → Generate result

---

## Tech Stack

**Backend**: Python 3.11, FastAPI, SAM 2.1, Replicate API  
**Frontend**: Vue 3, Vite  
**Acceleration**: MPS (Apple Silicon)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/TooTKK/Spatial_Shift.git
cd Spatial_Shift
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Install SAM2

```bash
pip install -e ./sam2
```

### 4. Download Model

Download SAM 2.1 Hiera Large (~856MB) and place it in `backend/checkpoints/`:

```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 5. Configure Environment

Create `backend/.env`:

```bash
REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxx
```

---

## Quick Start

**Backend:**
```bash
cd backend
python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

API runs at `http://localhost:8000`  
Frontend runs at `http://localhost:5173`

---

## Usage

### Web Interface

1. Open browser: `http://localhost:5173`
2. Upload a room image
3. Click on the furniture you want to move (red marker appears)
4. Click on the target position (blue marker appears)
5. Wait for processing (AI generates the result)
6. View the final image with furniture at the new position

### API Endpoints

**Full Pipeline:**
```bash
curl -X POST "http://localhost:8000/full_pipeline" \
  -F "file=@room.jpg" \
  -F "segment_x=200" \
  -F "segment_y=300" \
  -F "place_x=500" \
  -F "place_y=400" \
  -F "use_ai=false"
```

Returns:
```json
{
  "image_id": "xxx",
  "furniture_bbox": [x1, y1, x2, y2],
  "final_image": "base64_encoded_image"
}
```

---

## Architecture

```
User uploads image
    ↓
Click furniture position → SAM 2.1 segmentation
    ↓
Extract furniture mask
    ↓
Click target position → Inpainting + Alpha Blending
    ↓
Generate final image
```

---

## Project Structure

```
Spatial_Shift/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── sam.py                  # SAM 2.1 segmentation
│   ├── inpainting_cloud.py     # Replicate inpainting
│   ├── furniture_placement.py  # Alpha Blending placement
│   ├── checkpoints/
│   │   └── sam2.1_hiera_large.pt
│   └── output/
│       ├── furniture/          # Segmented furniture
│       ├── backgrounds/        # Cleaned backgrounds
│       └── placed/             # Final results
├── frontend/
│   └── src/
│       └── App.vue             # Main UI component
└── sam2/                       # SAM 2.1 source code
```

---

## Troubleshooting

### Backend fails to start

- Check if SAM model exists: `backend/checkpoints/sam2.1_hiera_large.pt`
- Verify SAM2 installation: `pip install -e ./sam2`
- Check Python version: `python --version` (requires 3.11+)

### API returns 500 error

- Check backend terminal for detailed error messages
- Verify `.env` file exists with correct API keys
- Ensure all dependencies are installed

### Frontend cannot connect

- Check if backend is running: `curl http://localhost:8000/`
- Verify `API_BASE` in `App.vue` is set to `http://localhost:8000`
- Check CORS configuration in backend

### Processing is slow

- Replicate API: ~30-60 seconds per request (cloud processing)
- Local processing (OpenCV fallback): ~2-5 seconds (lower quality)

---

## Development

### Running Tests

```bash
cd backend
python test_sam.py          # Test SAM segmentation
python test_placement.py    # Test furniture placement
python test_backend.py      # Backend health check
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Notes

- Floor detection automatically removes unwanted ground regions
- Coordinate scaling handles display-to-pixel conversion

---

**Built for HackIllinois 2025**
