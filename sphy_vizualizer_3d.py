"""
sphy_vizualizer_3d.py — Exotic Wave Field em 3D com py5
Lê sphy_frames.parquet, valida SHA256 e anima a onda como malha 3D.

Dependências:
    pip install py5 pandas pyarrow numpy
    Requer Java 17+ instalado e ambiente gráfico (não roda headless).

Controles:
    Mouse drag  → rotacionar a cena
    Scroll      → zoom
    SPACE       → pausar / retomar animação
    R           → reiniciar do frame 0
    Q / ESC     → fechar
"""

import numpy as np
import pandas as pd
import hashlib
import os
import sys
import py5

# ─────────────────────────────────────────────────────────────
# 1. CARREGAR E VALIDAR PARQUET
# ─────────────────────────────────────────────────────────────

PARQUET_PATH = "sphy_frames.parquet"

if not os.path.exists(PARQUET_PATH):
    print(f"\n  ERRO: '{PARQUET_PATH}' não encontrado.")
    print("  Execute sphy_gerador.py primeiro.\n")
    sys.exit(1)

print("=" * 58)
print("  SPHY VISUALIZER 3D — py5  |  Exotic Wave Field + SHA256")
print("=" * 58)
print(f"\n  Carregando: {PARQUET_PATH}")

df = pd.read_parquet(PARQUET_PATH)
TOTAL_FRAMES = len(df)
SHAPE_ROWS   = int(df["shape_rows"].iloc[0])   # eixo t  (100)
SHAPE_COLS   = int(df["shape_cols"].iloc[0])   # eixo x  (200)

print(f"  Frames : {TOTAL_FRAMES}  |  Grid : {SHAPE_ROWS} × {SHAPE_COLS}")

# ── Validação SHA256 ──────────────────────────────────────────
x_arr      = np.linspace(-10, 10, SHAPE_COLS)
t_arr      = np.linspace(0, 10,   SHAPE_ROWS)
X_grid, T_grid = np.meshgrid(x_arr, t_arr)

print("\n  Validando SHA256 de todos os frames (pode levar alguns segundos)...")

falhas   = 0
waves_3d = []   # lista de arrays (SHAPE_ROWS, SHAPE_COLS)

for idx, row in df.iterrows():
    t_off  = float(row["t_offset"])
    T_anim = T_grid + t_off
    wave   = (np.sin(2 * np.pi * 0.3 * X_grid - 2 * np.pi * 0.1 * T_anim)
              * np.exp(-0.05 * X_grid**2))

    sha_calc   = hashlib.sha256(wave.tobytes()).hexdigest()
    sha_stored = row["sha256"]

    if sha_calc != sha_stored:
        falhas += 1
        print(f"  ❌ Frame {idx}: hash divergente!")

    waves_3d.append(wave)

    pct = (idx + 1) / TOTAL_FRAMES * 100
    if (idx + 1) % max(1, TOTAL_FRAMES // 10) == 0 or idx == TOTAL_FRAMES - 1:
        print(f"  [{pct:5.1f}%] Frame {idx + 1:>5}/{TOTAL_FRAMES}")

SHA_STATUS = "✅ SHA256 OK" if falhas == 0 else f"❌ {falhas} frame(s) corrompido(s)"
print(f"\n  {SHA_STATUS}")
print("=" * 58)

# ─────────────────────────────────────────────────────────────
# 2. SUB-AMOSTRAGEM para performance 3D
#    Reduz a malha para ~40×80 vértices — suave e rápido
# ─────────────────────────────────────────────────────────────

STEP_T = max(1, SHAPE_ROWS  // 40)   # passos no eixo t
STEP_X = max(1, SHAPE_COLS  // 80)   # passos no eixo x

x_sub = x_arr[::STEP_X]
t_sub = t_arr[::STEP_T]
NX    = len(x_sub)
NT    = len(t_sub)

# Pré-processa todos os frames sub-amostrados
waves_sub = [w[::STEP_T, ::STEP_X] for w in waves_3d]

print(f"\n  Malha 3D : {NT} × {NX} vértices por frame — pronto!\n")

# ─────────────────────────────────────────────────────────────
# 3. ESTADO GLOBAL DA ANIMAÇÃO
# ─────────────────────────────────────────────────────────────

state = {
    "frame"     : 0,
    "paused"    : False,
    "rot_x"     : -0.45,   # ângulo inicial de inclinação
    "rot_y"     : 0.3,
    "drag_start": None,
    "zoom"      : 1.0,
}

# ─────────────────────────────────────────────────────────────
# 4. HELPERS DE COR — paleta "inferno" simples via GLSL-like math
# ─────────────────────────────────────────────────────────────

def inferno_color(t: float):
    """
    Aproximação da paleta inferno em (r, g, b) 0-255.
    t ∈ [0, 1]
    """
    t = float(np.clip(t, 0, 1))
    r = int(np.clip(255 * (0.8 * t**0.5 + 0.2 * t**3), 0, 255))
    g = int(np.clip(255 * (0.7 * t**2.5),               0, 255))
    b = int(np.clip(255 * (0.6 * (1 - t)**2 + 0.2 * t), 0, 255))
    return r, g, b

# ─────────────────────────────────────────────────────────────
# 5. py5 SKETCH
# ─────────────────────────────────────────────────────────────

W, H = 960, 620
AMP  = 60    # escala vertical da onda em pixels
SCALE_X = 8  # pixels por unidade x
SCALE_T = 6  # pixels por unidade t

def setup():
    py5.size(W, H, py5.P3D)
    py5.color_mode(py5.RGB, 255)
    py5.frame_rate(40)
    py5.hint(py5.ENABLE_DEPTH_TEST)

def draw():
    s = state

    # ── Fundo escuro ──────────────────────────────────────────
    py5.background(12, 10, 20)

    # ── Câmera / projeção ─────────────────────────────────────
    py5.push_matrix()

    py5.translate(W * 0.5, H * 0.52, -200 * s["zoom"])
    py5.rotate_x(s["rot_x"])
    py5.rotate_y(s["rot_y"])

    # Centralizar a malha na origem
    offset_x = -(NX - 1) * SCALE_X / 2
    offset_t = -(NT - 1) * SCALE_T / 2

    wave = waves_sub[s["frame"]]

    # ── Desenhar malha como triângulos (TRIANGLE_STRIP por linha) ──
    py5.no_stroke()
    for ti in range(NT - 1):
        py5.begin_shape(py5.TRIANGLE_STRIP)
        for xi in range(NX):
            for dti in [0, 1]:
                tti = ti + dti
                px  = offset_x + xi  * SCALE_X
                pz  = offset_t + tti * SCALE_T
                py_ = -wave[tti, xi] * AMP    # y invertido (py5: y cresce p/ baixo)

                # Cor pela amplitude normalizada [0,1]
                c_t = (wave[tti, xi] + 1) / 2
                r, g, b = inferno_color(c_t)

                # Iluminação simples: face superior mais brilhante
                bright = int(np.clip(120 + wave[tti, xi] * 135, 0, 255))
                py5.fill(
                    int(r * bright / 255),
                    int(g * bright / 255),
                    int(b * bright / 255),
                )
                py5.vertex(px, py_, pz)
        py5.end_shape()

    # ── Linhas de grade suaves ────────────────────────────────
    py5.stroke(60, 55, 80, 120)
    py5.stroke_weight(0.4)
    py5.no_fill()

    # Grade x
    for xi in range(0, NX, 4):
        py5.begin_shape()
        for ti in range(NT):
            px  = offset_x + xi * SCALE_X
            pz  = offset_t + ti * SCALE_T
            py_ = -wave[ti, xi] * AMP
            py5.vertex(px, py_, pz)
        py5.end_shape()

    # Grade t
    for ti in range(0, NT, 3):
        py5.begin_shape()
        for xi in range(NX):
            px  = offset_x + xi * SCALE_X
            pz  = offset_t + ti * SCALE_T
            py_ = -wave[ti, xi] * AMP
            py5.vertex(px, py_, pz)
        py5.end_shape()

    py5.pop_matrix()

    # ── HUD ───────────────────────────────────────────────────
    _draw_hud(s["frame"])

    # ── Avançar frame ─────────────────────────────────────────
    if not s["paused"]:
        s["frame"] = (s["frame"] + 1) % TOTAL_FRAMES


def _draw_hud(frame: int):
    """Overlay 2D com info e SHA256."""
    sha = df["sha256"].iloc[frame]
    paused_txt = "  ⏸ PAUSADO" if state["paused"] else ""

    py5.hint(py5.DISABLE_DEPTH_TEST)
    py5.camera()         # reseta para câmera 2D
    py5.no_lights()

    # Painel fundo
    py5.no_stroke()
    py5.fill(10, 8, 18, 200)
    py5.rect(0, 0, W, 54)
    py5.rect(0, H - 30, W, 30)

    # Textos
    py5.fill(230, 220, 255)
    py5.text_size(13)
    py5.text("SPHY Exotic Wave Field — 3D Visualizer", 14, 18)
    py5.text_size(11)
    py5.fill(160, 155, 200)
    py5.text(f"Frame {frame + 1}/{TOTAL_FRAMES}  |  {SHA_STATUS}{paused_txt}", 14, 36)

    py5.fill(80, 200, 120)
    py5.text_size(10)
    py5.text(f"SHA256: {sha}", 14, H - 10)

    py5.fill(120, 115, 160)
    py5.text("Drag: rotacionar  |  Scroll: zoom  |  SPACE: pausar  |  R: reiniciar  |  Q: sair",
             W // 2 - 220, H - 10)
    py5.hint(py5.ENABLE_DEPTH_TEST)


# ── Interação: mouse ──────────────────────────────────────────

def mouse_pressed():
    state["drag_start"] = (py5.mouse_x, py5.mouse_y)


def mouse_dragged():
    if state["drag_start"] is None:
        return
    dx = py5.mouse_x - state["drag_start"][0]
    dy = py5.mouse_y - state["drag_start"][1]
    state["rot_y"] += dx * 0.008
    state["rot_x"] += dy * 0.008
    state["drag_start"] = (py5.mouse_x, py5.mouse_y)


def mouse_released():
    state["drag_start"] = None


def mouse_wheel(event):
    state["zoom"] = float(np.clip(state["zoom"] + event.get_count() * 0.08, 0.3, 4.0))


# ── Interação: teclado ────────────────────────────────────────

def key_pressed():
    k = py5.key
    if k == " ":
        state["paused"] = not state["paused"]
    elif k in ("r", "R"):
        state["frame"] = 0
    elif k in ("q", "Q"):
        py5.exit_sketch()


# ─────────────────────────────────────────────────────────────
# 6. RUN
# ─────────────────────────────────────────────────────────────

py5.run_sketch(
    sketch_functions={
        "setup"         : setup,
        "draw"          : draw,
        "mouse_pressed" : mouse_pressed,
        "mouse_dragged" : mouse_dragged,
        "mouse_released": mouse_released,
        "mouse_wheel"   : mouse_wheel,
        "key_pressed"   : key_pressed,
    },
    block=True,
)
