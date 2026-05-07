"""
sphy_hydrogen_atom.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Visualizador: Átomo de Hidrogênio  ←  sphy_frames.parquet

A amplitude da onda exótica (campo escalar calculado no
parquet e assinado em SHA-256) modula diretamente:

  • Raio orbital do elétron      ← amplitude média do frame
  • Velocidade angular           ← frequência instantânea da onda
  • Densidade da nuvem eletrônica← distribuição espacial da onda
  • Cor do elétron               ← paleta inferno mapeada ao frame

A física é universal: a mesma equação de onda que descreve
o campo escalar exótico governa a função de onda do elétron.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import hashlib, os, sys

# ═══════════════════════════════════════════════════════
# 1 — CARREGAR PARQUET
# ═══════════════════════════════════════════════════════

PARQUET = "sphy_frames.parquet"

if not os.path.exists(PARQUET):
    print(f"\n  ❌  '{PARQUET}' não encontrado.")
    print("  Execute sphy_gerador.py primeiro.\n")
    sys.exit(1)

print("━" * 58)
print("  SPHY HYDROGEN ATOM  ←  sphy_frames.parquet")
print("━" * 58)

df = pd.read_parquet(PARQUET)
TOTAL      = len(df)
ROWS       = int(df["shape_rows"].iloc[0])
COLS       = int(df["shape_cols"].iloc[0])

x_arr      = np.linspace(-10, 10, COLS)
t_arr      = np.linspace(0,  10,  ROWS)
X_g, T_g   = np.meshgrid(x_arr, t_arr)

print(f"  Frames : {TOTAL}  |  Grid : {ROWS}×{COLS}")

# ═══════════════════════════════════════════════════════
# 2 — PRÉ-COMPUTAR WAVES + VALIDAR SHA256
# ═══════════════════════════════════════════════════════

print("\n  Calculando waves e validando SHA-256 ...")

waves      = []
sha_ok_all = True
sha_flags  = []

for i, row in df.iterrows():
    t_off   = float(row["t_offset"])
    T_anim  = T_g + t_off
    w       = (np.sin(2*np.pi*0.3*X_g - 2*np.pi*0.1*T_anim)
               * np.exp(-0.05 * X_g**2))
    waves.append(w)

    h_calc  = hashlib.sha256(w.tobytes()).hexdigest()
    h_store = row["sha256"]
    ok      = (h_calc == h_store)
    sha_flags.append(ok)
    if not ok:
        sha_ok_all = False

    if (i+1) % max(1, TOTAL//8) == 0 or i == TOTAL-1:
        pct = (i+1)/TOTAL*100
        sym = "✓" if ok else "✗ FALHA"
        print(f"  [{pct:5.1f}%]  frame {i+1:>4}/{TOTAL}  {sym}  {h_calc[:20]}…")

SHA_LABEL = ("✅  SHA-256 íntegro — todos os frames validados"
             if sha_ok_all else f"❌  SHA-256 FALHOU em {sha_flags.count(False)} frame(s)")
print(f"\n  {SHA_LABEL}")
print("━" * 58)

# ═══════════════════════════════════════════════════════
# 3 — EXTRAIR PARÂMETROS FÍSICOS DO PARQUET POR FRAME
# ═══════════════════════════════════════════════════════
# Estes valores VÊM do parquet — a onda modula o átomo.

amp_mean  = np.array([w.mean()          for w in waves])   # amplitude média
amp_std   = np.array([w.std()           for w in waves])   # dispersão (→ raio)
amp_max   = np.array([np.abs(w).max()   for w in waves])   # pico absoluto
energy    = np.array([(w**2).mean()     for w in waves])   # energia da onda

# Normaliza para faixas físicas interessantes
def norm01(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)

radius_n  = 0.35 + 0.55 * norm01(amp_std)    # raio orbital  [0.35, 0.90]
omega_n   = 0.8  + 1.4  * norm01(energy)     # vel. angular  [0.8, 2.2] rad/frame
color_n   = norm01(amp_max)                  # cor paleta inferno

# ═══════════════════════════════════════════════════════
# 4 — HELPERS VISUAIS
# ═══════════════════════════════════════════════════════

def inferno(t):
    """Paleta inferno → (r,g,b) ∈ [0,1]"""
    t = np.clip(t, 0, 1)
    r = np.clip(1.0  * t**0.45, 0, 1)
    g = np.clip(0.85 * t**2.2,  0, 1)
    b = np.clip(0.7  * (1-t)**1.8 + 0.15*t, 0, 1)
    return (r, g, b)

def electron_color(t):
    r, g, b = inferno(t)
    return (r, g, b, 1.0)

def orbit_color(t, alpha=0.35):
    r, g, b = inferno(t)
    return (r, g, b, alpha)

# ═══════════════════════════════════════════════════════
# 5 — FIGURA
# ═══════════════════════════════════════════════════════

BG  = "#080810"
FG  = "#d8d4f0"
ACC = "#ff6e3a"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "text.color"       : FG,
    "axes.labelcolor"  : FG,
    "xtick.color"      : FG,
    "ytick.color"      : FG,
    "axes.edgecolor"   : "#2a2840",
    "font.family"      : "monospace",
})

fig = plt.figure(figsize=(14, 8), facecolor=BG)
fig.canvas.manager.set_window_title("SPHY — Hydrogen Atom ← parquet")

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    left=0.04, right=0.98,
    top=0.93,  bottom=0.07,
    wspace=0.38, hspace=0.55,
)

ax_atom  = fig.add_subplot(gs[:, 0:2])   # átomo (grande, esquerda)
ax_wave  = fig.add_subplot(gs[0, 2])     # onda do frame
ax_sha   = fig.add_subplot(gs[1, 2])     # barra SHA / parâmetros
ax_hist  = fig.add_subplot(gs[2, 2])     # histórico de energia

for ax in [ax_atom, ax_wave, ax_sha, ax_hist]:
    ax.set_facecolor(BG)

# ── Átomo ────────────────────────────────────────────────────
ax_atom.set_xlim(-1.1, 1.1)
ax_atom.set_ylim(-1.1, 1.1)
ax_atom.set_aspect("equal")
ax_atom.set_title("Átomo de Hidrogênio  ←  parquet field", color=FG,
                  fontsize=11, pad=8, fontfamily="monospace")
ax_atom.set_xticks([])
ax_atom.set_yticks([])
for sp in ax_atom.spines.values():
    sp.set_edgecolor("#1a1830")

# Nuvem eletrônica (pontos de probabilidade)
N_CLOUD   = 280
rng       = np.random.default_rng(42)
CLOUD_ANG = rng.uniform(0, 2*np.pi, N_CLOUD)
CLOUD_RAD = rng.rayleigh(0.42, N_CLOUD)
cx0 = CLOUD_RAD * np.cos(CLOUD_ANG)
cy0 = CLOUD_RAD * np.sin(CLOUD_ANG)
cloud_sc = ax_atom.scatter(cx0, cy0, s=1.5, alpha=0.0,
                           c=[BG]*N_CLOUD, zorder=1)

# Órbita (círculo tracejado)
theta_circ = np.linspace(0, 2*np.pi, 300)
orbit_line, = ax_atom.plot([], [], lw=0.8, linestyle="--",
                           color="#3a3560", alpha=0.6, zorder=2)

# Trilha do elétron
TRAIL     = 40
trail_x   = np.zeros(TRAIL)
trail_y   = np.zeros(TRAIL)
trail_sc  = ax_atom.scatter(trail_x, trail_y, s=0, zorder=3)

# Elétron
elec_sc  = ax_atom.scatter([0.5], [0], s=140, zorder=5,
                            color="#ff9040", edgecolors="#fff5e0",
                            linewidths=1.2)

# Próton
proton_sc = ax_atom.scatter([0], [0], s=320, zorder=4,
                             color="#e03030", edgecolors="#ffaaaa",
                             linewidths=1.5)
ax_atom.text(0, 0, "p⁺", color="white", fontsize=7,
             ha="center", va="center", zorder=6, fontweight="bold")

# Linha elétron→próton (ligação)
bond_line, = ax_atom.plot([], [], lw=0.6, color="#4a4570",
                           alpha=0.5, zorder=2)

# Texto de parâmetros no átomo
param_txt = ax_atom.text(
    -1.07, -1.05,
    "", color="#9890c8", fontsize=7.5,
    va="bottom", fontfamily="monospace", zorder=7,
)

# Símbolo quântico
ax_atom.text(1.07, 1.05, "H  n=1  1s", color="#5550a0",
             fontsize=8, ha="right", va="top", fontstyle="italic")

# ── Onda do frame ────────────────────────────────────────────
ax_wave.set_title("Campo Escalar (parquet)", color=FG, fontsize=8, pad=4)
ax_wave.set_xlabel("posição x", color=FG, fontsize=7)
ax_wave.set_ylabel("amplitude", color=FG, fontsize=7)
ax_wave.tick_params(labelsize=6)
ax_wave.set_xlim(-10, 10)
ax_wave.set_ylim(-1.1, 1.1)
ax_wave.axhline(0, color="#2a2840", lw=0.6)
wave_line, = ax_wave.plot([], [], lw=1.2, color="#ff7030")
wave_fill  = ax_wave.fill_between([], [], alpha=0)
x_center_line = ax_wave.axvline(0, color="#60ffaa", lw=0.8,
                                 linestyle=":", alpha=0.6)

# ── Painel SHA ───────────────────────────────────────────────
ax_sha.set_xlim(0, 1)
ax_sha.set_ylim(0, 1)
ax_sha.set_xticks([])
ax_sha.set_yticks([])
ax_sha.set_title("Assinatura SHA-256", color=FG, fontsize=8, pad=4)
for sp in ax_sha.spines.values():
    sp.set_edgecolor("#1a1830")

sha_color_ok  = "#30ff80"
sha_color_bad = "#ff3030"

sha_bar   = ax_sha.barh([0.7], [1.0], height=0.18,
                         color=sha_color_ok if sha_ok_all else sha_color_bad,
                         alpha=0.85)[0]
sha_label = ax_sha.text(0.5, 0.70, "", color="white", fontsize=6.5,
                         ha="center", va="center",
                         fontfamily="monospace")
sha_status_txt = ax_sha.text(0.5, 0.42, SHA_LABEL[:38],
                               color=sha_color_ok if sha_ok_all else sha_color_bad,
                               fontsize=6.5, ha="center", va="center",
                               fontfamily="monospace")
sha_frame_txt  = ax_sha.text(0.5, 0.22, "", color="#9890c8",
                               fontsize=6.5, ha="center", va="center",
                               fontfamily="monospace")
sha_param_txt  = ax_sha.text(0.5, 0.06, "", color="#6860a8",
                               fontsize=6, ha="center", va="center",
                               fontfamily="monospace")

# ── Histórico de energia ─────────────────────────────────────
ax_hist.set_title("Energia da Onda (parquet → átomo)", color=FG,
                   fontsize=8, pad=4)
ax_hist.set_xlabel("frame", color=FG, fontsize=7)
ax_hist.set_ylabel("E  ∝  ⟨ψ²⟩", color=FG, fontsize=7)
ax_hist.tick_params(labelsize=6)
ax_hist.set_xlim(0, TOTAL)
ax_hist.set_ylim(energy.min()*0.98, energy.max()*1.02)
ax_hist.plot(energy, color="#2a2840", lw=0.6, alpha=0.4)   # fundo
hist_cursor = ax_hist.axvline(0, color=ACC, lw=1.0, alpha=0.8)
hist_dot,   = ax_hist.plot([], [], "o", ms=5,
                            color=ACC, markeredgecolor="#fff5e0",
                            markeredgewidth=0.7)

# ── Título global ────────────────────────────────────────────
fig.text(0.5, 0.977,
         "FÍSICA UNIVERSAL  —  sphy_frames.parquet  →  Átomo de Hidrogênio",
         ha="center", va="top", color=FG, fontsize=11, fontfamily="monospace",
         fontweight="bold")
fig.text(0.5, 0.960,
         "A onda exótica calculada no parquet modula a função de onda do elétron em tempo real",
         ha="center", va="top", color="#6860a8", fontsize=8, fontstyle="italic",
         fontfamily="monospace")

# ═══════════════════════════════════════════════════════
# 6 — ESTADO DA ANIMAÇÃO
# ═══════════════════════════════════════════════════════

angle = [0.0]   # ângulo acumulado do elétron

# ═══════════════════════════════════════════════════════
# 7 — FUNÇÃO DE ATUALIZAÇÃO
# ═══════════════════════════════════════════════════════

def update(frame):
    w   = waves[frame]
    sha = df["sha256"].iloc[frame]
    ok  = sha_flags[frame]

    R   = radius_n[frame]
    Om  = omega_n[frame]
    col = color_n[frame]
    en  = energy[frame]

    # ── Avança ângulo (velocidade vem do parquet) ─────────────
    angle[0] += Om * 0.07     # fator de escala para velocidade visual
    theta     = angle[0]

    ex = R * np.cos(theta)
    ey = R * np.sin(theta)

    # ── Órbita ───────────────────────────────────────────────
    ox = R * np.cos(theta_circ)
    oy = R * np.sin(theta_circ)
    orbit_line.set_data(ox, oy)
    orbit_line.set_color(orbit_color(col, 0.35))

    # ── Trilha do elétron ────────────────────────────────────
    trail_x[:-1] = trail_x[1:]
    trail_y[:-1] = trail_y[1:]
    trail_x[-1]  = ex
    trail_y[-1]  = ey

    sizes  = np.linspace(2, 55, TRAIL)
    alphas = np.linspace(0.04, 0.85, TRAIL)
    ec     = electron_color(col)
    colors = [(ec[0], ec[1], ec[2], a) for a in alphas]
    trail_sc.set_offsets(np.c_[trail_x, trail_y])
    trail_sc.set_sizes(sizes)
    trail_sc.set_color(colors)

    # ── Elétron ───────────────────────────────────────────────
    ec_full = electron_color(col)
    elec_sc.set_offsets([[ex, ey]])
    elec_sc.set_color([ec_full])

    # ── Ligação p→e ───────────────────────────────────────────
    bond_line.set_data([0, ex], [0, ey])

    # ── Nuvem de probabilidade (modulada pelo std da onda) ────
    cloud_r  = np.clip(CLOUD_RAD * R / 0.42, 0, 1.05)
    cx_new   = cloud_r * np.cos(CLOUD_ANG)
    cy_new   = cloud_r * np.sin(CLOUD_ANG)
    cloud_sc.set_offsets(np.c_[cx_new, cy_new])
    cloud_sc.set_sizes(np.full(N_CLOUD, 1.8))
    cloud_alp = np.clip(0.08 + 0.18 * norm01(en) * np.ones(N_CLOUD), 0, 0.30)
    cloud_colors = [(ec_full[0], ec_full[1], ec_full[2], a) for a in cloud_alp]
    cloud_sc.set_color(cloud_colors)

    # ── Texto de parâmetros no átomo ─────────────────────────
    param_txt.set_text(
        f"r = {R:.3f} a₀   ω = {Om:.3f} rad/u\n"
        f"E = {en:.5f}   |ψ|²∝ parquet"
    )

    # ── Onda do frame ─────────────────────────────────────────
    w_slice = w[ROWS//2, :]          # fatia central do campo
    wave_line.set_data(x_arr, w_slice)
    ax_wave.set_title(f"Campo Escalar  —  frame {frame+1}", color=FG,
                       fontsize=8, pad=4)
    # Preenche área
    for coll in ax_wave.collections:
        coll.remove()
    ax_wave.fill_between(x_arr, w_slice, alpha=0.15,
                          color=inferno(col))

    # Cursor na posição x do elétron
    x_center_line.set_xdata([ex * 10])   # re-escala: átomo ∈[-1,1], onda ∈[-10,10]

    # ── SHA-256 ───────────────────────────────────────────────
    sha_short = sha[:32] + "…"
    sha_label.set_text(sha_short)
    bar_col   = sha_color_ok if ok else sha_color_bad
    sha_bar.set_color(bar_col)
    sha_frame_txt.set_text(f"frame {frame+1}/{TOTAL}  {'✓ OK' if ok else '✗ FALHA'}")
    sha_param_txt.set_text(f"r={R:.3f}a₀  ω={Om:.3f}  E={en:.5f}")

    # ── Histórico ─────────────────────────────────────────────
    hist_cursor.set_xdata([frame])
    hist_dot.set_data([frame], [en])
    hist_dot.set_color(inferno(col))

    return (orbit_line, trail_sc, elec_sc, bond_line, cloud_sc,
            wave_line, sha_label, sha_bar, sha_frame_txt,
            sha_param_txt, hist_cursor, hist_dot, param_txt)


# ═══════════════════════════════════════════════════════
# 8 — RODAR
# ═══════════════════════════════════════════════════════

ani = FuncAnimation(
    fig, update,
    frames=TOTAL,
    interval=30,     # ~33 FPS
    blit=False,      # False para permitir ax.set_title dinâmico
)

plt.show()
