# Fingerprint of God

### Dual 3D Mathematical Beauty Visualization

A stunning split-screen visualization featuring two of the most beautiful objects in mathematics:

| LEFT | RIGHT |
|------|-------|
| **Mandelbulb Fractal** | **Riemann Zeta Function** |
| Biomechanical temple from another dimension | The "Fingerprint of God" on the critical line |

Both rendered in **real-time 3D** with GPU-accelerated ray marching.

---

## ✨ Features

### 🔮 Mandelbulb (Left Panel)
- Ray-marched 3D fractal with 8th power iteration
- Iridescent cosmic color palette
- Soft shadows and ambient occlusion
- Pearlescent surface reflections

### ∞ Riemann Zeta (Right Panel)
- 3D visualization of ζ(½ + it) on the critical line
- Animated flowing curve showing the zeta function's path
- **Zeros glow bright orange-red** where |ζ(s)| → 0
- Golden/bronze iridescent coloring

### 🎬 Shared Features
- Real-time GPU rendering at 60fps
- Synchronized camera rotation
- Elegant golden divider
- Cinematic post-processing

---

## 🎮 Controls

| Input | Action |
|-------|--------|
| **Left Mouse Drag** | Rotate both views |
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
git clone https://github.com/3sholding/fingerprint-of-God.git
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

---

## 📐 The Mathematics

### Mandelbulb
The Mandelbulb extends the Mandelbrot iteration z → z² + c to 3D using spherical coordinates:

```
z → z^8 + c
```

The 8th power creates the iconic bulbous, organic structure that looks like architecture from an alien civilization.

### Riemann Zeta Function

The Riemann zeta function:

```
ζ(s) = Σ(n=1 to ∞) 1/n^s
```

On the **critical line** Re(s) = ½, the function traces a complex path through 3D space. The **zeros** — where ζ(½ + it) = 0 — appear as special points and are marked with glowing orange-red.

The **Riemann Hypothesis** (unsolved, $1M prize) states all non-trivial zeros lie on this line.

---

## 🎨 Color Palettes

| Visualization | Palette |
|--------------|---------|
| Mandelbulb | Cosmic teals → Nebula purples → Aurora greens |
| Riemann Zeta | Golden bronze → Warm copper → Orange at zeros |

Both use **iridescent/pearlescent** effects that shift color based on viewing angle.

---

## 📁 Project Structure

```
fingerprint-of-God/
├── f_o_g.py           # Main dual visualization
├── requirements.txt   # Python dependencies  
├── README.md          # This file
└── output/            # Screenshots saved here
```

---

## 👤 Author

**Prof. Shahab Anbarjafari**  
*Designer & Developer*

**3S Holding OÜ**  
Tartu, Estonia

---

## ⚖️ Copyright & Legal Notice

**© 2025 3S Holding OÜ. All Rights Reserved.**

This software, including but not limited to its source code, documentation, visual outputs, algorithms, and associated intellectual property, is the exclusive property of **3S Holding OÜ**, a company registered and incorporated in Tartu, Estonia.

### Terms of Use

1. **Proprietary Software**: This software is proprietary and confidential. Unauthorized copying, modification, distribution, or use of this software, in whole or in part, is strictly prohibited.

2. **No License Granted**: No license, express or implied, is granted to any party under any patent, copyright, trademark, or other intellectual property right of 3S Holding OÜ, except as explicitly stated in a separate written agreement.

3. **Restricted Use**: This software may only be used with the express written permission of 3S Holding OÜ. Any unauthorized use, reproduction, or distribution may result in civil and criminal penalties.

4. **No Warranty**: THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.

5. **Limitation of Liability**: IN NO EVENT SHALL 3S HOLDING OÜ BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE.

### Contact

For licensing inquiries, permissions, or legal matters, please contact:

**3S Holding OÜ**  
Tartu, Estonia  
European Union

---

<p align="center">
  <i>"The zeros of the zeta function are the fingerprint of God."</i>
  <br><br>
  <b>Two infinities, one screen.</b>
  <br><br>
  <sub>© 2025 3S Holding OÜ, Tartu, Estonia. All Rights Reserved.</sub>
</p>
