# =============================================================================
# MANDELBULB HERO — Interactive 3D Fractal Viewer
# =============================================================================
# 
# Real-time ray-marched Mandelbulb with mouse rotation
# GPU-accelerated rendering using OpenGL shaders
#
# Controls:
#   - Left Mouse Drag: Rotate camera
#   - Scroll Wheel: Zoom in/out
#   - Space: Toggle auto-rotation
#   - S: Save screenshot
#   - R: Reset camera
#   - ESC: Quit
#
# Designer & Developer: Prof. Shahab Anbarjafari
# Organization: 3S Holding OÜ
# =============================================================================

import moderngl
import pygame
from pygame.locals import *
import numpy as np
from datetime import datetime
import os

# =============================================================================
# GLSL SHADER CODE
# =============================================================================

VERTEX_SHADER = """
#version 330 core
in vec2 in_position;
out vec2 fragCoord;

void main() {
    fragCoord = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core

uniform vec2 iResolution;
uniform float iTime;
uniform vec2 iMouse;
uniform float iZoom;
uniform int iAutoRotate;

out vec4 fragColor;

// =============================================================================
// CONFIGURATION
// =============================================================================

#define MAX_STEPS 128
#define MAX_DIST 50.0
#define SURF_DIST 0.0008
#define MANDELBULB_POWER 8.0
#define MANDELBULB_ITERATIONS 12

// =============================================================================
// ELEGANT COLOR PALETTES
// =============================================================================

// Attempt to create smooth, elegant gradients like cosmic auroras
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

// Iridescent cosmic palette - deep teals, magentas, golds
vec3 cosmicPalette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.00, 0.10, 0.20);  // Teal to gold shift
    return palette(t, a, b, c, d);
}

// Aurora borealis palette
vec3 auroraPalette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 0.5);
    vec3 d = vec3(0.80, 0.90, 0.30);  // Green to magenta
    return palette(t, a, b, c, d);
}

// Nebula palette - deep purples and cyans
vec3 nebulaPalette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(2.0, 1.0, 0.0);
    vec3 d = vec3(0.50, 0.20, 0.25);  // Purple to cyan
    return palette(t, a, b, c, d);
}

// Pearlescent/opal palette
vec3 opalPalette(float t) {
    vec3 col = vec3(0.0);
    col += 0.5 + 0.5 * cos(6.28318 * (t + vec3(0.0, 0.1, 0.2)));
    col *= 0.5 + 0.5 * cos(6.28318 * (t * 2.0 + vec3(0.3, 0.2, 0.1)));
    return col;
}

// =============================================================================
// MANDELBULB DISTANCE ESTIMATOR
// =============================================================================

vec3 mandelbulbDE(vec3 pos) {
    vec3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    float trap = 1e10;
    float trapY = 0.0;
    
    float power = MANDELBULB_POWER;
    
    for (int i = 0; i < MANDELBULB_ITERATIONS; i++) {
        r = length(z);
        if (r > 2.0) break;
        
        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);
        
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        
        float zr = pow(r, power);
        theta *= power;
        phi *= power;
        
        z = zr * vec3(
            sin(theta) * cos(phi),
            sin(theta) * sin(phi),
            cos(theta)
        );
        z += pos;
        
        // Multiple orbit traps for richer coloring
        float newTrap = length(z);
        if (newTrap < trap) {
            trap = newTrap;
            trapY = z.y;
        }
    }
    
    float dist = 0.5 * log(r) * r / dr;
    return vec3(dist, trap, trapY);
}

// =============================================================================
// RAY MARCHING
// =============================================================================

vec4 rayMarch(vec3 ro, vec3 rd) {
    float d = 0.0;
    float trap = 0.0;
    float trapY = 0.0;
    int steps = 0;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * d;
        vec3 result = mandelbulbDE(p);
        float dist = result.x;
        trap = result.y;
        trapY = result.z;
        
        if (dist < SURF_DIST) {
            steps = i;
            break;
        }
        if (d > MAX_DIST) {
            steps = MAX_STEPS;
            break;
        }
        
        d += dist * 0.6;
        steps = i;
    }
    
    return vec4(d, trap, trapY, float(steps));
}

// =============================================================================
// NORMAL & LIGHTING
// =============================================================================

vec3 getNormal(vec3 p) {
    float eps = 0.0001;
    vec2 h = vec2(eps, 0.0);
    return normalize(vec3(
        mandelbulbDE(p + h.xyy).x - mandelbulbDE(p - h.xyy).x,
        mandelbulbDE(p + h.yxy).x - mandelbulbDE(p - h.yxy).x,
        mandelbulbDE(p + h.yyx).x - mandelbulbDE(p - h.yyx).x
    ));
}

float ambientOcclusion(vec3 p, vec3 n) {
    float ao = 0.0;
    float scale = 1.0;
    
    for (int i = 0; i < 5; i++) {
        float dist = 0.01 + 0.15 * float(i);
        float d = mandelbulbDE(p + n * dist).x;
        ao += (dist - d) * scale;
        scale *= 0.5;
    }
    
    return clamp(1.0 - ao * 2.5, 0.0, 1.0);
}

float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    
    for (int i = 0; i < 32; i++) {
        float h = mandelbulbDE(ro + rd * t).x;
        res = min(res, k * h / t);
        t += clamp(h, 0.01, 0.08);
        if (h < 0.0005 || t > maxt) break;
    }
    
    return clamp(res, 0.0, 1.0);
}

// =============================================================================
// CAMERA
// =============================================================================

mat3 setCamera(vec3 ro, vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    return mat3(cu, cv, cw);
}

// =============================================================================
// MAIN RENDER
// =============================================================================

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
    
    // Camera control
    vec2 mouse = iMouse / iResolution;
    
    float angle, elevation;
    if (iAutoRotate == 1) {
        angle = iTime * 0.25;
        elevation = 0.25 + sin(iTime * 0.15) * 0.2;
    } else {
        angle = mouse.x * 6.283185;
        elevation = (mouse.y - 0.5) * 3.14159 * 0.9;
    }
    
    float radius = iZoom;
    
    vec3 ro = vec3(
        radius * cos(angle) * cos(elevation),
        radius * sin(elevation),
        radius * sin(angle) * cos(elevation)
    );
    vec3 ta = vec3(0.0, 0.0, 0.0);
    
    mat3 ca = setCamera(ro, ta, 0.0);
    vec3 rd = ca * normalize(vec3(uv, 1.8));
    
    // Ray march
    vec4 result = rayMarch(ro, rd);
    float d = result.x;
    float trap = result.y;
    float trapY = result.z;
    float steps = result.w;
    
    // =========================================================================
    // ELEGANT NEBULA BACKGROUND
    // =========================================================================
    
    // Multi-layered nebula gradient
    vec3 col = vec3(0.0);
    
    // Base deep space
    col = mix(vec3(0.01, 0.01, 0.02), vec3(0.02, 0.01, 0.04), uv.y * 0.5 + 0.5);
    
    // Nebula clouds
    float nebula1 = sin(uv.x * 3.0 + iTime * 0.1) * cos(uv.y * 2.0 - iTime * 0.05);
    float nebula2 = cos(uv.x * 5.0 - iTime * 0.08) * sin(uv.y * 4.0 + iTime * 0.12);
    col += vec3(0.05, 0.02, 0.08) * (nebula1 * 0.5 + 0.5) * 0.3;
    col += vec3(0.02, 0.04, 0.08) * (nebula2 * 0.5 + 0.5) * 0.2;
    
    // Subtle star field
    vec2 starUV = floor(uv * 300.0);
    float star = fract(sin(dot(starUV, vec2(12.9898, 78.233))) * 43758.5453);
    star = smoothstep(0.98, 1.0, star);
    col += star * vec3(0.8, 0.9, 1.0) * 0.4;
    
    // Distant galaxy glow
    float galaxyDist = length(uv - vec2(0.3, 0.2));
    col += vec3(0.1, 0.05, 0.15) * exp(-galaxyDist * 4.0) * 0.5;
    
    // =========================================================================
    // SURFACE RENDERING
    // =========================================================================
    
    if (d < MAX_DIST) {
        vec3 p = ro + rd * d;
        vec3 n = getNormal(p);
        vec3 viewDir = -rd;
        
        // ----- ELEGANT MULTI-LIGHT SETUP -----
        
        // Key light - warm golden
        vec3 lightDir1 = normalize(vec3(0.6, 0.8, -0.4));
        vec3 lightCol1 = vec3(1.0, 0.9, 0.7) * 1.2;
        
        // Fill light - cool cyan
        vec3 lightDir2 = normalize(vec3(-0.7, 0.3, 0.6));
        vec3 lightCol2 = vec3(0.4, 0.7, 1.0) * 0.8;
        
        // Rim light - magenta accent
        vec3 lightDir3 = normalize(vec3(0.0, -0.5, -0.8));
        vec3 lightCol3 = vec3(1.0, 0.3, 0.6) * 0.6;
        
        // Diffuse
        float diff1 = max(dot(n, lightDir1), 0.0);
        float diff2 = max(dot(n, lightDir2), 0.0);
        float diff3 = max(dot(n, lightDir3), 0.0);
        
        // Specular (Blinn-Phong)
        vec3 h1 = normalize(lightDir1 + viewDir);
        vec3 h2 = normalize(lightDir2 + viewDir);
        float spec1 = pow(max(dot(n, h1), 0.0), 64.0);
        float spec2 = pow(max(dot(n, h2), 0.0), 32.0);
        
        // AO and shadows
        float ao = ambientOcclusion(p, n);
        float shadow = softShadow(p + n * 0.01, lightDir1, 0.02, 5.0, 16.0);
        
        // ----- IRIDESCENT COLOR CALCULATION -----
        
        // Multiple factors for rich, varied coloring
        float colorFactor1 = trap * 1.2;
        float colorFactor2 = trapY * 0.8 + 0.5;
        float colorFactor3 = length(p.xz) * 0.3;
        float colorFactor4 = p.y * 0.5 + 0.5;
        
        // Blend multiple palettes for iridescent effect
        vec3 col1 = cosmicPalette(colorFactor1 + iTime * 0.02);
        vec3 col2 = nebulaPalette(colorFactor2 - iTime * 0.015);
        vec3 col3 = auroraPalette(colorFactor3 + colorFactor4);
        
        // Fresnel-based palette blending (view-dependent iridescence)
        float fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0);
        
        // Blend colors based on surface properties
        vec3 baseColor = mix(col1, col2, smoothstep(0.3, 0.7, colorFactor1));
        baseColor = mix(baseColor, col3, fresnel * 0.5);
        
        // Add pearlescent sheen
        vec3 pearlShift = opalPalette(dot(n, viewDir) * 2.0 + trap);
        baseColor = mix(baseColor, pearlShift, fresnel * 0.3);
        
        // Desaturate slightly in shadows for elegance
        float luma = dot(baseColor, vec3(0.299, 0.587, 0.114));
        baseColor = mix(vec3(luma), baseColor, 0.7 + shadow * 0.3);
        
        // ----- FINAL LIGHTING COMPOSITION -----
        
        // Ambient (subtle, tinted)
        vec3 ambient = vec3(0.08, 0.06, 0.12) * ao;
        
        // Diffuse contribution
        vec3 diffuse = vec3(0.0);
        diffuse += diff1 * lightCol1 * shadow;
        diffuse += diff2 * lightCol2 * 0.6;
        diffuse += diff3 * lightCol3 * 0.4;
        
        // Combine
        col = baseColor * (ambient + diffuse * ao);
        
        // Specular highlights (colored)
        col += spec1 * lightCol1 * shadow * 0.5;
        col += spec2 * lightCol2 * 0.25;
        
        // Elegant rim lighting
        float rim = pow(1.0 - max(dot(n, viewDir), 0.0), 4.0);
        vec3 rimColor = mix(vec3(0.3, 0.5, 0.9), vec3(0.9, 0.4, 0.7), sin(trap * 5.0) * 0.5 + 0.5);
        col += rim * rimColor * 0.5;
        
        // Subsurface scattering (warm inner glow)
        float sss = pow(max(dot(viewDir, -lightDir1), 0.0), 2.0);
        col += sss * baseColor * vec3(1.2, 0.8, 0.6) * 0.15;
        
        // Depth-based color shift (atmospheric perspective)
        float depthFade = 1.0 - exp(-d * 0.06);
        vec3 atmosphereColor = vec3(0.15, 0.1, 0.25);
        col = mix(col, atmosphereColor, depthFade * 0.6);
        
        // Subtle glow from ray march complexity
        float complexity = steps / float(MAX_STEPS);
        col += vec3(0.1, 0.05, 0.15) * complexity * 0.4;
    }
    
    // =========================================================================
    // CINEMATIC POST-PROCESSING
    // =========================================================================
    
    // Subtle chromatic aberration at edges
    float chromaDist = length(uv) * 0.02;
    col.r *= 1.0 + chromaDist;
    col.b *= 1.0 - chromaDist * 0.5;
    
    // Elegant vignette
    float vignette = 1.0 - 0.4 * pow(length(uv * 0.75), 2.5);
    col *= vignette;
    
    // Soft bloom on bright areas
    float luminance = dot(col, vec3(0.299, 0.587, 0.114));
    float bloom = smoothstep(0.5, 1.0, luminance);
    col += col * bloom * 0.15;
    
    // Subtle film grain for organic feel
    float grain = fract(sin(dot(gl_FragCoord.xy + iTime, vec2(12.9898, 78.233))) * 43758.5453);
    col += (grain - 0.5) * 0.015;
    
    // Color grading - lift shadows, subtle teal/orange
    col = pow(col, vec3(0.95));  // Lift
    col = mix(col, col * vec3(1.05, 0.98, 0.95), 0.3);  // Warm highlights
    
    // Gamma correction
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
    
    fragColor = vec4(col, 1.0);
}
"""


# =============================================================================
# MANDELBULB VIEWER CLASS
# =============================================================================

class MandelbulbViewer:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.running = True
        
        # Camera state
        self.mouse_x = width / 2
        self.mouse_y = height / 2
        self.zoom = 3.0
        self.auto_rotate = True
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Mandelbulb Hero — Prof. Shahab Anbarjafari | 3S Holding OÜ")
        
        # Request OpenGL 3.3 core profile
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        
        # Create window
        self.screen = pygame.display.set_mode(
            (width, height), 
            DOUBLEBUF | OPENGL | RESIZABLE
        )
        
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        
        # Create shader program
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        
        # Set up full-screen quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')
        
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_position')
        
        # Get uniform locations
        self.u_resolution = self.prog['iResolution']
        self.u_time = self.prog['iTime']
        self.u_mouse = self.prog['iMouse']
        self.u_zoom = self.prog['iZoom']
        self.u_auto_rotate = self.prog['iAutoRotate']
        
        # Set initial values
        self.u_resolution.value = (float(width), float(height))
        self.u_zoom.value = self.zoom
        self.u_auto_rotate.value = 1 if self.auto_rotate else 0
        
        # Clock for timing
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()
        
        # Font for UI
        self.font = pygame.font.Font(None, 24)
        
        print("\n" + "="*60)
        print("  MANDELBULB HERO — Interactive 3D Fractal")
        print("="*60)
        print("  Designer: Prof. Shahab Anbarjafari")
        print("  Organization: 3S Holding OÜ")
        print("="*60)
        print("\n  Controls:")
        print("    Left Mouse Drag : Rotate camera")
        print("    Scroll Wheel    : Zoom in/out")
        print("    Space           : Toggle auto-rotation")
        print("    S               : Save screenshot")
        print("    R               : Reset camera")
        print("    ESC             : Quit")
        print("\n  Generating Mandelbulb in real-time...")
        print("="*60 + "\n")
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
                
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_SPACE:
                    self.auto_rotate = not self.auto_rotate
                    self.u_auto_rotate.value = 1 if self.auto_rotate else 0
                    print(f"  Auto-rotation: {'ON' if self.auto_rotate else 'OFF'}")
                elif event.key == K_s:
                    self.save_screenshot()
                elif event.key == K_r:
                    self.reset_camera()
                    
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.dragging = True
                    self.last_mouse_pos = event.pos
                    if self.auto_rotate:
                        self.auto_rotate = False
                        self.u_auto_rotate.value = 0
                        # Initialize mouse position from current auto-rotate angle
                        t = (pygame.time.get_ticks() - self.start_time) / 1000.0
                        self.mouse_x = (t * 0.3 / 6.283185) * self.width
                        self.mouse_y = (0.3 + np.sin(t * 0.2) * 0.2 + 0.5) * self.height
                elif event.button == 4:  # Scroll up
                    self.zoom = max(1.5, self.zoom - 0.2)
                    self.u_zoom.value = self.zoom
                elif event.button == 5:  # Scroll down
                    self.zoom = min(8.0, self.zoom + 0.2)
                    self.u_zoom.value = self.zoom
                    
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
                    
            elif event.type == MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.mouse_x += dx
                    self.mouse_y += dy
                    self.mouse_y = max(0, min(self.height, self.mouse_y))
                    self.last_mouse_pos = event.pos
                    
            elif event.type == VIDEORESIZE:
                self.width, self.height = event.size
                self.ctx.viewport = (0, 0, self.width, self.height)
                self.u_resolution.value = (float(self.width), float(self.height))
    
    def reset_camera(self):
        self.zoom = 3.0
        self.mouse_x = self.width / 2
        self.mouse_y = self.height / 2
        self.auto_rotate = True
        self.u_zoom.value = self.zoom
        self.u_auto_rotate.value = 1
        print("  Camera reset")
    
    def save_screenshot(self):
        if not os.path.exists("output"):
            os.makedirs("output")
        
        # Read pixels from framebuffer
        data = self.ctx.screen.read(components=3)
        img = pygame.image.fromstring(data, (self.width, self.height), 'RGB')
        img = pygame.transform.flip(img, False, True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/Mandelbulb_Hero_{timestamp}.png"
        pygame.image.save(img, filename)
        print(f"  ✓ Screenshot saved: {filename}")
    
    def render(self):
        # Update time
        current_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
        self.u_time.value = current_time
        
        # Update mouse position
        self.u_mouse.value = (self.mouse_x, self.mouse_y)
        
        # Clear and render
        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
        # Swap buffers
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(60)  # 60 FPS cap
        
        pygame.quit()
        print("\n  Goodbye! 🌌\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║              MANDELBULB HERO — 3D Fractal Viewer             ║")
    print("  ╠═══════════════════════════════════════════════════════════════╣")
    print("  ║  Designer & Developer: Prof. Shahab Anbarjafari              ║")
    print("  ║  Organization: 3S Holding OÜ                                  ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print("="*70)
    
    viewer = MandelbulbViewer(1280, 720)
    viewer.run()


if __name__ == "__main__":
    main()
