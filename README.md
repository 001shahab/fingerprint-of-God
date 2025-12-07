# Mandelbulb Hero

### Interactive 3D Fractal Viewer

A stunning real-time ray-marched **Mandelbulb fractal** that opens in a window and lets you rotate it interactively. GPU-accelerated for silky smooth 60fps rendering.

---

## ✨ Features

- **Real-time Ray Marching** — GPU-accelerated rendering using OpenGL shaders
- **Interactive Camera** — Click and drag to rotate, scroll to zoom
- **Auto-rotation Mode** — Watch the fractal spin automatically
- **Cinematic Lighting** — Multi-light setup with soft shadows and ambient occlusion
- **Screenshot Capture** — Save high-quality images anytime

---

## 🎮 Controls

| Input | Action |
|-------|--------|
| **Left Mouse Drag** | Rotate camera |
| **Scroll Wheel** | Zoom in/out |
| **Space** | Toggle auto-rotation |
| **S** | Save screenshot |
| **R** | Reset camera |
| **ESC** | Quit |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenGL 3.3+ compatible graphics card

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fingerprint-of-God.git
cd fingerprint-of-God

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python f_o_g.py
```

A window will open showing the Mandelbulb fractal generating in real-time. Use your mouse to rotate it!

---

## 📐 What is a Mandelbulb?

The **Mandelbulb** is a 3D fractal discovered in 2009 by Daniel White and Paul Nylander. It extends the famous Mandelbrot set into three dimensions using spherical coordinates.

The formula iterates:

```
z → z^n + c
```

Where the power operation in 3D spherical coordinates creates the stunning organic, biomechanical structures you see — like temples from an alien civilization.

---

## 🎨 Technical Details

- **Rendering**: Ray marching with sphere tracing
- **Lighting**: Phong model with soft shadows and ambient occlusion
- **Coloring**: Orbit trap + position-based HSV mapping
- **Post-processing**: Vignette, bloom, gamma correction

---

## 📁 Project Structure

```
fingerprint-of-God/
├── f_o_g.py           # Main viewer application
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── output/            # Screenshots saved here
```

---

## 👤 Author

**Prof. Shahab Anbarjafari**  
*Designer & Developer*

**3S Holding OÜ**

---

## 📄 License

MIT License

---

<p align="center">
  <i>"The Mandelbulb looks like biomechanical temples from an alien civilization."</i>
</p>
