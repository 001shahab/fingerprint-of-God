# =============================================================================
# FINGERPRINT OF GOD — Dual 3D Visualization
# =============================================================================
#
# Copyright (c) 2025 3S Holding OÜ. All Rights Reserved.
# Tartu, Estonia, European Union
#
# PROPRIETARY AND CONFIDENTIAL
#
# This software and its source code are the exclusive property of 3S Holding OÜ.
# Unauthorized copying, modification, distribution, or use of this software,
# in whole or in part, is strictly prohibited without the express written
# permission of 3S Holding OÜ.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED. IN NO EVENT SHALL 3S HOLDING OÜ BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE.
#
# =============================================================================
#
# Split-screen interactive viewer featuring:
#   LEFT:  Ray-marched 3D Mandelbulb fractal
#   RIGHT: Animated 3D Riemann Zeta function on the critical line
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
# Organization: 3S Holding OÜ, Tartu, Estonia
#
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

#define PI 3.14159265359
#define TAU 6.28318530718

// =============================================================================
// ELEGANT COLOR PALETTES
// =============================================================================

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(TAU * (c * t + d));
}

vec3 cosmicPalette(float t) {
    return palette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.00, 0.10, 0.20));
}

vec3 auroraPalette(float t) {
    return palette(t, vec3(0.5), vec3(0.5), vec3(1.0, 1.0, 0.5), vec3(0.80, 0.90, 0.30));
}

vec3 nebulaPalette(float t) {
    return palette(t, vec3(0.5), vec3(0.5), vec3(2.0, 1.0, 0.0), vec3(0.50, 0.20, 0.25));
}

vec3 opalPalette(float t) {
    vec3 col = 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.1, 0.2)));
    col *= 0.5 + 0.5 * cos(TAU * (t * 2.0 + vec3(0.3, 0.2, 0.1)));
    return col;
}

vec3 zetaPalette(float t) {
    return palette(t, vec3(0.5), vec3(0.5), vec3(1.0, 0.7, 0.4), vec3(0.0, 0.15, 0.20));
}

vec3 goldenPalette(float t) {
    return palette(t, vec3(0.5, 0.4, 0.3), vec3(0.5, 0.4, 0.2), vec3(1.0), vec3(0.0, 0.1, 0.2));
}

// =============================================================================
// MANDELBULB DISTANCE ESTIMATOR
// =============================================================================

#define MB_MAX_STEPS 100
#define MB_MAX_DIST 40.0
#define MB_SURF_DIST 0.001
#define MB_POWER 8.0
#define MB_ITERATIONS 10

vec3 mandelbulbDE(vec3 pos) {
    vec3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    float trap = 1e10;
    float trapY = 0.0;
    
    for (int i = 0; i < MB_ITERATIONS; i++) {
        r = length(z);
        if (r > 2.0) break;
        
        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);
        
        dr = pow(r, MB_POWER - 1.0) * MB_POWER * dr + 1.0;
        
        float zr = pow(r, MB_POWER);
        theta *= MB_POWER;
        phi *= MB_POWER;
        
        z = zr * vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        z += pos;
        
        float newTrap = length(z);
        if (newTrap < trap) {
            trap = newTrap;
            trapY = z.y;
        }
    }
    
    return vec3(0.5 * log(r) * r / dr, trap, trapY);
}

// =============================================================================
// RIEMANN ZETA FUNCTION (APPROXIMATION)
// =============================================================================

// Complex multiplication
vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex division
vec2 cdiv(vec2 a, vec2 b) {
    float denom = dot(b, b);
    return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / denom;
}

// Complex exponential
vec2 cexp(vec2 z) {
    return exp(z.x) * vec2(cos(z.y), sin(z.y));
}

// Complex power: n^(-s) where s = 0.5 + it
vec2 cpow_inv(float n, vec2 s) {
    float logn = log(n);
    // n^(-s) = exp(-s * log(n))
    vec2 exponent = vec2(-s.x * logn, -s.y * logn);
    return cexp(exponent);
}

// Riemann Zeta on critical line: ζ(0.5 + it)
// Using Dirichlet series with convergence acceleration
vec2 zeta_critical(float t) {
    vec2 s = vec2(0.5, t);
    vec2 sum = vec2(0.0);
    
    // Dirichlet eta function (alternating series converges better)
    // η(s) = Σ(-1)^(n-1) / n^s = (1 - 2^(1-s)) * ζ(s)
    
    const int N = 50;
    float sign = 1.0;
    
    for (int n = 1; n <= N; n++) {
        vec2 term = cpow_inv(float(n), s);
        sum += sign * term;
        sign = -sign;
    }
    
    // Convert eta to zeta: ζ(s) = η(s) / (1 - 2^(1-s))
    vec2 one_minus_s = vec2(0.5, -t);
    vec2 two_pow = cexp(vec2(log(2.0) * one_minus_s.x, log(2.0) * one_minus_s.y));
    vec2 denom = vec2(1.0, 0.0) - two_pow;
    
    return cdiv(sum, denom);
}

// =============================================================================
// 3D ZETA VISUALIZATION - Distance Field
// =============================================================================

#define ZETA_MAX_STEPS 80
#define ZETA_MAX_DIST 30.0
#define ZETA_SURF_DIST 0.015

// Distance to the 3D zeta curve (rendered as a tube/ribbon)
vec4 zetaCurveDE(vec3 p, float time) {
    float minDist = 1e10;
    float colorParam = 0.0;
    float nearestT = 0.0;
    vec3 nearestPoint = vec3(0.0);
    
    // Sample the zeta function along the critical line
    // Map 3D space to find closest point on the curve
    
    // The curve: (t, Re(ζ), Im(ζ)) scaled and animated
    float tOffset = time * 2.0;
    
    // Search along the curve
    for (int i = 0; i < 60; i++) {
        float t = float(i) * 0.8 + tOffset;
        vec2 z = zeta_critical(t);
        
        // 3D position of the curve point
        // x = imaginary part of input (t), y = Re(zeta), z = Im(zeta)
        float scale = 0.4;
        vec3 curvePoint = vec3(
            (t - tOffset) * 0.1 - 2.0,  // Spread along x
            z.x * scale,                  // Real part
            z.y * scale                   // Imaginary part
        );
        
        float d = length(p - curvePoint);
        if (d < minDist) {
            minDist = d;
            colorParam = t * 0.1;
            nearestT = t;
            nearestPoint = curvePoint;
        }
    }
    
    // Tube radius varies with magnitude of zeta (zeros become thin)
    vec2 zetaVal = zeta_critical(nearestT);
    float mag = length(zetaVal);
    float tubeRadius = 0.03 + 0.05 * mag;
    
    return vec4(minDist - tubeRadius, colorParam, mag, nearestT);
}

// Alternative: Zeta as a height field / surface
float zetaSurfaceDE(vec3 p, float time) {
    // Domain: xz plane maps to complex plane
    // Height (y) represents |ζ(s)| or phase
    
    float tOffset = time * 0.5;
    
    // Map position to complex plane
    float re = 0.5;  // Critical line
    float im = p.x * 5.0 + 10.0 + tOffset;  // Imaginary part
    
    vec2 z = zeta_critical(im);
    float mag = length(z);
    float phase = atan(z.y, z.x);
    
    // Surface height based on log magnitude
    float targetY = log(mag + 0.1) * 0.3;
    
    // Distance to surface
    float surfaceDist = p.y - targetY;
    
    // Add some thickness
    return abs(surfaceDist) - 0.02;
}

// Combined Zeta 3D scene
vec4 zetaSceneDE(vec3 p, float time) {
    // Render both curve and surface
    vec4 curve = zetaCurveDE(p, time);
    
    // Flowing ribbon/surface beneath
    float surface = zetaSurfaceDE(p - vec3(0.0, -0.8, 0.0), time);
    
    if (curve.x < surface) {
        return curve;
    } else {
        return vec4(surface, p.x * 0.2 + time * 0.1, 0.5, 0.0);
    }
}

// =============================================================================
// RAY MARCHING
// =============================================================================

vec4 rayMarchMandelbulb(vec3 ro, vec3 rd) {
    float d = 0.0;
    float trap = 0.0, trapY = 0.0;
    int steps = 0;
    
    for (int i = 0; i < MB_MAX_STEPS; i++) {
        vec3 p = ro + rd * d;
        vec3 result = mandelbulbDE(p);
        float dist = result.x;
        trap = result.y;
        trapY = result.z;
        
        if (dist < MB_SURF_DIST) { steps = i; break; }
        if (d > MB_MAX_DIST) { steps = MB_MAX_STEPS; break; }
        
        d += dist * 0.6;
        steps = i;
    }
    
    return vec4(d, trap, trapY, float(steps));
}

vec4 rayMarchZeta(vec3 ro, vec3 rd, float time) {
    float d = 0.0;
    vec4 info = vec4(0.0);
    int steps = 0;
    
    for (int i = 0; i < ZETA_MAX_STEPS; i++) {
        vec3 p = ro + rd * d;
        vec4 result = zetaSceneDE(p, time);
        float dist = result.x;
        
        if (dist < ZETA_SURF_DIST) {
            info = result;
            steps = i;
            break;
        }
        if (d > ZETA_MAX_DIST) {
            steps = ZETA_MAX_STEPS;
            break;
        }
        
        d += dist * 0.8;
        info = result;
        steps = i;
    }
    
    return vec4(d, info.y, info.z, float(steps));
}

// =============================================================================
// NORMALS
// =============================================================================

vec3 getMandelbulbNormal(vec3 p) {
    float eps = 0.0001;
    vec2 h = vec2(eps, 0.0);
    return normalize(vec3(
        mandelbulbDE(p + h.xyy).x - mandelbulbDE(p - h.xyy).x,
        mandelbulbDE(p + h.yxy).x - mandelbulbDE(p - h.yxy).x,
        mandelbulbDE(p + h.yyx).x - mandelbulbDE(p - h.yyx).x
    ));
}

vec3 getZetaNormal(vec3 p, float time) {
    float eps = 0.001;
    vec2 h = vec2(eps, 0.0);
    return normalize(vec3(
        zetaSceneDE(p + h.xyy, time).x - zetaSceneDE(p - h.xyy, time).x,
        zetaSceneDE(p + h.yxy, time).x - zetaSceneDE(p - h.yxy, time).x,
        zetaSceneDE(p + h.yyx, time).x - zetaSceneDE(p - h.yyx, time).x
    ));
}

// =============================================================================
// AMBIENT OCCLUSION
// =============================================================================

float mandelbulbAO(vec3 p, vec3 n) {
    float ao = 0.0;
    float scale = 1.0;
    for (int i = 0; i < 5; i++) {
        float dist = 0.01 + 0.12 * float(i);
        ao += (dist - mandelbulbDE(p + n * dist).x) * scale;
        scale *= 0.5;
    }
    return clamp(1.0 - ao * 2.5, 0.0, 1.0);
}

float zetaAO(vec3 p, vec3 n, float time) {
    float ao = 0.0;
    float scale = 1.0;
    for (int i = 0; i < 4; i++) {
        float dist = 0.02 + 0.15 * float(i);
        ao += (dist - zetaSceneDE(p + n * dist, time).x) * scale;
        scale *= 0.5;
    }
    return clamp(1.0 - ao * 2.0, 0.0, 1.0);
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
// RENDER MANDELBULB (LEFT SIDE)
// =============================================================================

vec3 renderMandelbulb(vec2 uv, float angle, float elevation, float zoom) {
    vec3 ro = vec3(
        zoom * cos(angle) * cos(elevation),
        zoom * sin(elevation),
        zoom * sin(angle) * cos(elevation)
    );
    vec3 ta = vec3(0.0);
    mat3 ca = setCamera(ro, ta, 0.0);
    vec3 rd = ca * normalize(vec3(uv, 1.8));
    
    vec4 result = rayMarchMandelbulb(ro, rd);
    float d = result.x;
    float trap = result.y;
    float trapY = result.z;
    float steps = result.w;
    
    // Background
    vec3 col = mix(vec3(0.01, 0.01, 0.03), vec3(0.04, 0.02, 0.08), uv.y * 0.5 + 0.5);
    
    // Stars
    vec2 starUV = floor(uv * 200.0);
    float star = fract(sin(dot(starUV, vec2(12.9898, 78.233))) * 43758.5453);
    col += smoothstep(0.98, 1.0, star) * 0.3;
    
    if (d < MB_MAX_DIST) {
        vec3 p = ro + rd * d;
        vec3 n = getMandelbulbNormal(p);
        vec3 v = -rd;
        
        // Lighting
        vec3 l1 = normalize(vec3(0.6, 0.8, -0.4));
        vec3 l2 = normalize(vec3(-0.7, 0.3, 0.6));
        
        float diff1 = max(dot(n, l1), 0.0);
        float diff2 = max(dot(n, l2), 0.0);
        float spec = pow(max(dot(n, normalize(l1 + v)), 0.0), 64.0);
        float ao = mandelbulbAO(p, n);
        
        // Colors
        float cf1 = trap * 1.2;
        float cf2 = trapY * 0.8 + 0.5;
        float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.0);
        
        vec3 col1 = cosmicPalette(cf1 + iTime * 0.02);
        vec3 col2 = nebulaPalette(cf2);
        vec3 baseColor = mix(col1, col2, smoothstep(0.3, 0.7, cf1));
        baseColor = mix(baseColor, opalPalette(dot(n, v) * 2.0 + trap), fresnel * 0.3);
        
        col = baseColor * (0.1 + diff1 * vec3(1.0, 0.9, 0.7) * 0.6 + diff2 * vec3(0.4, 0.6, 1.0) * 0.3) * ao;
        col += spec * 0.4;
        col += fresnel * vec3(0.3, 0.4, 0.8) * 0.4;
        
        // Fog
        col = mix(vec3(0.02, 0.02, 0.05), col, exp(-d * 0.08));
    }
    
    return col;
}

// =============================================================================
// RENDER RIEMANN ZETA (RIGHT SIDE)
// =============================================================================

vec3 renderZeta(vec2 uv, float angle, float elevation, float zoom, float time) {
    vec3 ro = vec3(
        zoom * 1.5 * cos(angle) * cos(elevation),
        zoom * 0.8 * sin(elevation) + 0.3,
        zoom * 1.5 * sin(angle) * cos(elevation)
    );
    vec3 ta = vec3(0.0, 0.0, 0.0);
    mat3 ca = setCamera(ro, ta, 0.0);
    vec3 rd = ca * normalize(vec3(uv, 1.5));
    
    vec4 result = rayMarchZeta(ro, rd, time);
    float d = result.x;
    float colorParam = result.y;
    float mag = result.z;
    float steps = result.w;
    
    // Deep space background with golden tint
    vec3 col = mix(vec3(0.02, 0.01, 0.01), vec3(0.06, 0.03, 0.08), uv.y * 0.5 + 0.5);
    
    // Nebula effect
    float neb = sin(uv.x * 4.0 + time * 0.2) * cos(uv.y * 3.0);
    col += vec3(0.04, 0.02, 0.06) * (neb * 0.5 + 0.5) * 0.3;
    
    // Stars
    vec2 starUV = floor(uv * 150.0);
    float star = fract(sin(dot(starUV, vec2(12.9898, 78.233))) * 43758.5453);
    col += smoothstep(0.97, 1.0, star) * vec3(1.0, 0.9, 0.7) * 0.4;
    
    if (d < ZETA_MAX_DIST) {
        vec3 p = ro + rd * d;
        vec3 n = getZetaNormal(p, time);
        vec3 v = -rd;
        
        // Elegant lighting
        vec3 l1 = normalize(vec3(0.5, 1.0, -0.3));
        vec3 l2 = normalize(vec3(-0.6, 0.2, 0.7));
        vec3 l3 = normalize(vec3(0.0, -1.0, 0.0));
        
        float diff1 = max(dot(n, l1), 0.0);
        float diff2 = max(dot(n, l2), 0.0);
        float spec1 = pow(max(dot(n, normalize(l1 + v)), 0.0), 48.0);
        float spec2 = pow(max(dot(n, normalize(l2 + v)), 0.0), 32.0);
        float ao = zetaAO(p, n, time);
        
        // Golden/bronze palette for zeta
        vec3 baseColor = zetaPalette(colorParam);
        vec3 goldAccent = goldenPalette(colorParam * 0.5 + mag);
        
        // Zeros glow special color (where mag is low)
        float zeroGlow = exp(-mag * 5.0);
        vec3 zeroColor = vec3(1.0, 0.3, 0.1);  // Bright orange-red at zeros
        baseColor = mix(baseColor, zeroColor, zeroGlow * 0.8);
        
        // Iridescence
        float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.0);
        baseColor = mix(baseColor, goldAccent, fresnel * 0.4);
        
        // Combine lighting
        vec3 ambient = vec3(0.08, 0.06, 0.04);
        col = baseColor * (ambient + 
            diff1 * vec3(1.0, 0.85, 0.6) * 0.7 + 
            diff2 * vec3(0.5, 0.6, 1.0) * 0.4) * ao;
        
        // Specular
        col += spec1 * vec3(1.0, 0.9, 0.7) * 0.5;
        col += spec2 * vec3(0.6, 0.7, 1.0) * 0.3;
        
        // Rim light
        col += fresnel * vec3(0.8, 0.5, 0.2) * 0.4;
        
        // Extra glow at zeros
        col += zeroGlow * zeroColor * 0.5;
        
        // Fog
        col = mix(vec3(0.03, 0.02, 0.04), col, exp(-d * 0.1));
        
        // Glow from complexity
        col += vec3(0.08, 0.05, 0.02) * (steps / float(ZETA_MAX_STEPS)) * 0.3;
    }
    
    return col;
}

// =============================================================================
// MAIN
// =============================================================================

void main() {
    vec2 fragCoordNorm = gl_FragCoord.xy / iResolution.xy;
    
    // Camera control
    vec2 mouse = iMouse / iResolution;
    float angle, elevation;
    
    if (iAutoRotate == 1) {
        angle = iTime * 0.2;
        elevation = 0.25 + sin(iTime * 0.12) * 0.15;
    } else {
        angle = mouse.x * TAU;
        elevation = (mouse.y - 0.5) * PI * 0.9;
    }
    
    float zoom = iZoom;
    vec3 col;
    
    // Divider position (center)
    float divider = 0.5;
    float dividerWidth = 0.003;
    
    if (fragCoordNorm.x < divider - dividerWidth) {
        // LEFT SIDE: Mandelbulb
        vec2 uv = (gl_FragCoord.xy - vec2(iResolution.x * 0.25, iResolution.y * 0.5)) / iResolution.y * 2.0;
        col = renderMandelbulb(uv, angle, elevation, zoom);
        
    } else if (fragCoordNorm.x > divider + dividerWidth) {
        // RIGHT SIDE: Riemann Zeta
        vec2 uv = (gl_FragCoord.xy - vec2(iResolution.x * 0.75, iResolution.y * 0.5)) / iResolution.y * 2.0;
        col = renderZeta(uv, angle * 0.7 + 0.5, elevation * 0.8, zoom * 0.8, iTime);
        
    } else {
        // DIVIDER: Elegant golden line
        float glow = exp(-abs(fragCoordNorm.x - divider) * 500.0);
        col = vec3(0.8, 0.6, 0.2) * glow;
        col += vec3(1.0, 0.9, 0.7) * glow * glow;
    }
    
    // === POST-PROCESSING ===
    
    // Vignette
    vec2 uvScreen = fragCoordNorm - 0.5;
    float vignette = 1.0 - 0.3 * dot(uvScreen, uvScreen);
    col *= vignette;
    
    // Subtle film grain
    float grain = fract(sin(dot(gl_FragCoord.xy + iTime, vec2(12.9898, 78.233))) * 43758.5453);
    col += (grain - 0.5) * 0.01;
    
    // Color grading
    col = pow(col, vec3(0.95));
    
    // Gamma
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
    
    fragColor = vec4(col, 1.0);
}
"""


# =============================================================================
# MANDELBULB VIEWER CLASS
# =============================================================================

class MandelbulbViewer:
    def __init__(self, width=1440, height=810):
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
        pygame.display.set_caption("Fingerprint of God — Mandelbulb & Riemann Zeta | Prof. Shahab Anbarjafari | 3S Holding OÜ")
        
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
        
        print("\n" + "="*70)
        print("  ╔════════════════════════════════════════════════════════════════╗")
        print("  ║          FINGERPRINT OF GOD — Dual 3D Visualization           ║")
        print("  ╠════════════════════════════════════════════════════════════════╣")
        print("  ║  LEFT:  Mandelbulb Fractal (Biomechanical Temple)              ║")
        print("  ║  RIGHT: Riemann Zeta ζ(½+it) (Fingerprint of God)              ║")
        print("  ╠════════════════════════════════════════════════════════════════╣")
        print("  ║  Designer & Developer: Prof. Shahab Anbarjafari                ║")
        print("  ║  Organization: 3S Holding OÜ                                   ║")
        print("  ╚════════════════════════════════════════════════════════════════╝")
        print("="*70)
        print("\n  Controls:")
        print("    Left Mouse Drag : Rotate both views")
        print("    Scroll Wheel    : Zoom in/out")
        print("    Space           : Toggle auto-rotation")
        print("    S               : Save screenshot")
        print("    R               : Reset camera")
        print("    ESC             : Quit")
        print("\n  Rendering in real-time...")
        print("="*70 + "\n")
    
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
                if event.button == 1:
                    self.dragging = True
                    self.last_mouse_pos = event.pos
                    if self.auto_rotate:
                        self.auto_rotate = False
                        self.u_auto_rotate.value = 0
                        t = (pygame.time.get_ticks() - self.start_time) / 1000.0
                        self.mouse_x = (t * 0.2 / 6.283185) * self.width
                        self.mouse_y = (0.25 + np.sin(t * 0.12) * 0.15 + 0.5) * self.height
                elif event.button == 4:
                    self.zoom = max(1.5, self.zoom - 0.15)
                    self.u_zoom.value = self.zoom
                elif event.button == 5:
                    self.zoom = min(6.0, self.zoom + 0.15)
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
        
        data = self.ctx.screen.read(components=3)
        img = pygame.image.fromstring(data, (self.width, self.height), 'RGB')
        img = pygame.transform.flip(img, False, True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/Fingerprint_of_God_{timestamp}.png"
        pygame.image.save(img, filename)
        print(f"  ✓ Screenshot saved: {filename}")
    
    def render(self):
        current_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
        self.u_time.value = current_time
        self.u_mouse.value = (self.mouse_x, self.mouse_y)
        
        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()
        print("\n  Goodbye! 🌌\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║              FINGERPRINT OF GOD — 3D Visualization             ║")
    print("  ╠════════════════════════════════════════════════════════════════╣")
    print("  ║  Designer & Developer: Prof. Shahab Anbarjafari                ║")
    print("  ║  Organization: 3S Holding OÜ                                   ║")
    print("  ╚════════════════════════════════════════════════════════════════╝")
    print("="*70)
    
    viewer = MandelbulbViewer(1440, 810)
    viewer.run()


if __name__ == "__main__":
    main()
