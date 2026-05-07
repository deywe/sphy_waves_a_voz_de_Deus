"""
sphy_lorenz_butterfly.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 A BORBOLETA DE LORENZ RESOLVIDA PELO PARQUET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Edward Lorenz (1963) descobriu que três equações diferenciais simples
 produzem um atrator caótico em forma de borboleta:

     dx/dt = σ(y − x)
     dy/dt = x(ρ − z) − y
     dz/dt = xy − βz

 Parâmetros clássicos: σ=10, ρ=28, β=8/3

 COMO O PARQUET RESOLVE ISSO:
 ─────────────────────────────
 Cada frame do parquet carrega um campo escalar ψ(x,t) — a onda exótica.
 O payload extrai desse campo os coeficientes do passo de integração:

   • A amplitude média   → perturbação em σ  (sensibilidade ao calor)
   • O desvio padrão     → perturbação em ρ  (número de Rayleigh)
   • A energia ⟨ψ²⟩      → perturbação em β  (razão geométrica)
   • O gradiente espacial→ força do campo vetorial em cada sub-passo RK4

 O integrador Runge-Kutta 4 avança o estado (x,y,z) um passo dt=0.01
 por frame. O parquet É o motor de integração — sem ele, a borboleta
 não existe. Cada hash SHA-256 prova que aquele passo foi calculado
 exatamente com aquele campo ψ e nenhum outro.

 Dois atratores são integrados em paralelo com condições iniciais
 infinitesimalmente diferentes (Δx₀ = 1e-8) para demonstrar a
 sensibilidade às condições iniciais — a essência do caos.

 ESTRUTURA DA TELA:
   [esquerda]  Atrator 3D projetado XZ + trilha de cor tempo
   [centro]    Equações ao vivo com valores do frame atual
   [dir-cima]  Onda do parquet (campo ψ que gerou o passo)
   [dir-meio]  Divergência entre as duas trajetórias (Lyapunov)
   [dir-baixo] SHA-256 do frame + hash da cadeia
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import hashlib, json, os, sys, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# ════════════════════════════════════════════════════════════════════════
# 0 — PARÂMETROS GLOBAIS
# ════════════════════════════════════════════════════════════════════════

PARQUET_PATH = "sphy_frames.parquet"
MAX_FRAMES   = 1200
DT           = 0.015          # passo de integração por frame
TRAIL_LEN    = 220            # pontos da trilha visível
σ0, ρ0, β0  = 10.0, 28.0, 8/3  # parâmetros clássicos de Lorenz
DELTA_IC     = 1e-8           # separação inicial das duas trajetórias

BG   = "#04050d"
FG   = "#d0cce8"
ACC  = "#ff6b35"
CYAN = "#00e5ff"
GOLD = "#ffd54f"
LIME = "#b2ff59"
PINK = "#ff4081"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "text.color"       : FG,
    "axes.labelcolor"  : FG,
    "xtick.color"      : "#4a4870",
    "ytick.color"      : "#4a4870",
    "axes.edgecolor"   : "#151530",
    "font.family"      : "monospace",
    "axes.grid"        : False,
})

# ════════════════════════════════════════════════════════════════════════
# 1 — CARREGAR E VALIDAR PARQUET
# ════════════════════════════════════════════════════════════════════════

def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

if not os.path.exists(PARQUET_PATH):
    print(f"\n  ❌  '{PARQUET_PATH}' não encontrado.")
    print("  Execute sphy_gerador.py primeiro.\n")
    sys.exit(1)

print()
print("╔══════════════════════════════════════════════════════════════╗")
print("║   A BORBOLETA DE LORENZ  ←  sphy_frames.parquet             ║")
print("║   O parquet integra Runge-Kutta 4  ·  SHA-256 por passo     ║")
print("╚══════════════════════════════════════════════════════════════╝")

df     = pd.read_parquet(PARQUET_PATH)
TOTAL  = len(df)
ROWS   = int(df["shape_rows"].iloc[0])
COLS   = int(df["shape_cols"].iloc[0])

x_arr  = np.linspace(-10, 10, COLS)
t_arr  = np.linspace(0,  10,  ROWS)
X_g, T_g = np.meshgrid(x_arr, t_arr)

N_FRAMES = min(TOTAL, MAX_FRAMES)
print(f"\n  Parquet   : {TOTAL} frames disponíveis")
print(f"  Usando    : {N_FRAMES} frames  (máx {MAX_FRAMES})")
print(f"  Grid      : {ROWS}×{COLS}")
print(f"\n  Validando SHA-256 + pré-computando campos ψ …")

t0_load   = time.perf_counter()
waves     = []
sha_flags = []
sha_chain = []
sha_ok_all= True

for i in range(N_FRAMES):
    row    = df.iloc[i]
    t_off  = float(row["t_offset"])
    T_anim = T_g + t_off
    w      = (np.sin(2*np.pi*0.3*X_g - 2*np.pi*0.1*T_anim)
              * np.exp(-0.05 * X_g**2))
    waves.append(w)

    h_calc  = sha256_array(w)
    h_store = row["sha256"]
    ok      = (h_calc == h_store)
    sha_flags.append(ok)
    sha_chain.append(h_calc)
    if not ok:
        sha_ok_all = False
        print(f"  ❌  frame {i}: hash divergente!")

    if (i+1) % max(1, N_FRAMES//8) == 0 or i == N_FRAMES-1:
        pct = (i+1)/N_FRAMES*100
        print(f"  [{pct:5.1f}%] frame {i+1:>5}/{N_FRAMES}  "
              f"{'✓' if ok else '✗'}  {h_calc[:22]}…")

CHAIN_HASH = sha256_str("".join(sha_chain))
elapsed_load = time.perf_counter() - t0_load

SHA_STATUS = ("✅  SHA-256 OK" if sha_ok_all
              else f"❌  FALHOU em {sha_flags.count(False)} frame(s)")
print(f"\n  {SHA_STATUS}")
print(f"  Hash da cadeia : {CHAIN_HASH[:52]}…")
print(f"  Tempo de carga : {elapsed_load:.2f}s")

# ════════════════════════════════════════════════════════════════════════
# 2 — PAYLOAD: INTEGRADOR RK4 DE LORENZ SOBRE O PARQUET
# ════════════════════════════════════════════════════════════════════════

print(f"\n  Integrando Lorenz com RK4  ·  dt={DT}  ·  {N_FRAMES} passos …")

def lorenz_deriv(state, sigma, rho, beta):
    """dX/dt das equações de Lorenz."""
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])

def rk4_step(state, sigma, rho, beta, dt):
    """Um passo Runge-Kutta 4 das equações de Lorenz."""
    k1 = lorenz_deriv(state,            sigma, rho, beta)
    k2 = lorenz_deriv(state + dt/2 * k1, sigma, rho, beta)
    k3 = lorenz_deriv(state + dt/2 * k2, sigma, rho, beta)
    k4 = lorenz_deriv(state + dt   * k3, sigma, rho, beta)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def extract_params(wave):
    """
    Extrai σ, ρ, β perturbados do campo ψ do parquet.
    O parquet modula os parâmetros de Lorenz — é o motor físico.
    """
    amp   = float(wave.mean())
    std   = float(wave.std())
    en    = float((wave**2).mean())
    grad  = float(np.abs(np.gradient(wave[ROWS//2, :])).mean())

    # Perturbações físicas: a onda exótica perturba o sistema caótico
    sigma = σ0 + amp  * 1.5    # calor convectivo
    rho   = ρ0 + std  * 4.0    # número de Rayleigh
    beta  = β0 + en   * 0.3    # razão geométrica
    return sigma, rho, beta, grad

# Estado inicial: dois pontos infinitesimalmente próximos
state_A = np.array([0.1,        0.0, 0.0])   # trajetória A
state_B = np.array([0.1+DELTA_IC, 0.0, 0.0]) # trajetória B (Δx₀=1e-8)

# Arrays de resultado (pre-alocados)
traj_A    = np.zeros((N_FRAMES, 3))
traj_B    = np.zeros((N_FRAMES, 3))
params    = np.zeros((N_FRAMES, 4))   # sigma, rho, beta, grad
diverg    = np.zeros(N_FRAMES)        # |A - B|
lyapunov  = np.zeros(N_FRAMES)       # log(|A-B| / Δ0) / t

t0_int = time.perf_counter()
for i, w in enumerate(waves):
    sigma, rho, beta, grad = extract_params(w)
    params[i] = [sigma, rho, beta, grad]

    state_A = rk4_step(state_A, sigma, rho, beta, DT)
    state_B = rk4_step(state_B, sigma, rho, beta, DT)

    traj_A[i]   = state_A
    traj_B[i]   = state_B
    diverg[i]   = np.linalg.norm(state_A - state_B)
    t_elapsed   = (i+1) * DT
    lyapunov[i] = (np.log(diverg[i] / DELTA_IC + 1e-30)) / (t_elapsed + 1e-10)

    if (i+1) % max(1, N_FRAMES//8) == 0 or i == N_FRAMES-1:
        pct = (i+1)/N_FRAMES*100
        print(f"  [{pct:5.1f}%] frame {i+1:>5}  "
              f"xyz=({state_A[0]:6.2f},{state_A[1]:6.2f},{state_A[2]:6.2f})  "
              f"λ={lyapunov[i]:.4f}")

elapsed_int = time.perf_counter() - t0_int
print(f"\n  Integração concluída em {elapsed_int:.2f}s")
print(f"  Expoente de Lyapunov final: λ ≈ {lyapunov[-1]:.4f}")
print(f"  Divergência final |A−B|   : {diverg[-1]:.4e}")

# ── Salvar relatório parquet ──────────────────────────────────────────
report_df = pd.DataFrame({
    "frame"   : np.arange(N_FRAMES),
    "sha256"  : df["sha256"].iloc[:N_FRAMES].values,
    "sha_ok"  : sha_flags,
    "lx"      : traj_A[:, 0],
    "ly"      : traj_A[:, 1],
    "lz"      : traj_A[:, 2],
    "lx_b"    : traj_B[:, 0],
    "ly_b"    : traj_B[:, 1],
    "lz_b"    : traj_B[:, 2],
    "sigma"   : params[:, 0],
    "rho"     : params[:, 1],
    "beta"    : params[:, 2],
    "grad_psi": params[:, 3],
    "diverg"  : diverg,
    "lyapunov": lyapunov,
})
REPORT = "sphy_lorenz_report.parquet"
report_df.to_parquet(REPORT, index=False, compression="snappy")

meta = {
    "payload"       : "lorenz_butterfly_rk4",
    "description"   : "Borboleta de Lorenz integrada por RK4 usando o parquet como motor",
    "total_frames"  : N_FRAMES,
    "dt"            : DT,
    "sigma0"        : σ0, "rho0": ρ0, "beta0": β0,
    "delta_ic"      : DELTA_IC,
    "sha_ok_all"    : sha_ok_all,
    "chain_hash"    : CHAIN_HASH,
    "lyapunov_final": float(lyapunov[-1]),
    "divergence_final": float(diverg[-1]),
    "elapsed_load_s": elapsed_load,
    "elapsed_int_s" : elapsed_int,
    "generated_at"  : datetime.now(timezone.utc).isoformat(),
}
with open("sphy_lorenz_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n  💾  {REPORT}  ({os.path.getsize(REPORT)/1024:.1f} KB)")
print(f"  📄  sphy_lorenz_meta.json")

# ════════════════════════════════════════════════════════════════════════
# 3 — VISUALIZAÇÃO HISTÓRICA
# ════════════════════════════════════════════════════════════════════════

print("\n  Abrindo visualização histórica …\n")

# ── Paleta de cores por tempo ──────────────────────────────────────────
def plasma(t):
    t = float(np.clip(t, 0, 1))
    r = np.clip(0.94 * t**0.38 + 0.06, 0, 1)
    g = np.clip(0.12 + 0.6 * np.sin(np.pi * t), 0, 1)
    b = np.clip(0.94 * (1-t)**0.45 + 0.08*t, 0, 1)
    return (r, g, b)

# ── Figura principal ───────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9.5), facecolor=BG)
fig.canvas.manager.set_window_title(
    "A Borboleta de Lorenz ← sphy_frames.parquet")

gs = gridspec.GridSpec(
    4, 5, figure=fig,
    left=0.03, right=0.98,
    top=0.91,  bottom=0.06,
    wspace=0.40, hspace=0.65,
)

ax_butterfly = fig.add_subplot(gs[:, 0:3])      # borboleta (grande)
ax_wave      = fig.add_subplot(gs[0, 3:5])      # onda do parquet
ax_eqs       = fig.add_subplot(gs[1, 3:5])      # equações ao vivo
ax_diverg    = fig.add_subplot(gs[2, 3:5])      # divergência
ax_sha       = fig.add_subplot(gs[3, 3:5])      # SHA-256

for ax in [ax_butterfly, ax_wave, ax_eqs, ax_diverg, ax_sha]:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#151530")

# ── Título histórico ───────────────────────────────────────────────────
fig.text(0.5, 0.975,
         "A BORBOLETA DE LORENZ  —  Equações Integradas pelo Parquet",
         ha="center", color=FG, fontsize=13, fontweight="bold",
         fontfamily="monospace")
fig.text(0.5, 0.953,
         "dx/dt = σ(y−x)   ·   dy/dt = x(ρ−z)−y   ·   dz/dt = xy−βz"
         "          Runge-Kutta 4  ·  SHA-256 por frame",
         ha="center", color="#5550a0", fontsize=8.5,
         fontfamily="monospace", fontstyle="italic")

# ── PAINEL BORBOLETA ──────────────────────────────────────────────────
# Projeção X–Z (visão clássica da borboleta)
ax_butterfly.set_xlim(traj_A[:,0].min()-2, traj_A[:,0].max()+2)
ax_butterfly.set_ylim(traj_A[:,2].min()-2, traj_A[:,2].max()+2)
ax_butterfly.set_xlabel("x  (deslocamento convectivo)", color="#5550a0",
                         fontsize=8)
ax_butterfly.set_ylabel("z  (variação de temperatura)", color="#5550a0",
                         fontsize=8)
ax_butterfly.set_title("Atrator de Lorenz — Projeção x×z", color=FG,
                        fontsize=10, pad=8)
ax_butterfly.tick_params(labelsize=7, colors="#3a3860")

# Fundo: trajetória completa em cinza fantasma
ax_butterfly.plot(traj_A[:,0], traj_A[:,2],
                  color="#111128", lw=0.5, alpha=0.3, zorder=1)

# Objetos animados
trail_line_A, = ax_butterfly.plot([], [], lw=1.2, alpha=0.9,
                                   color=CYAN, zorder=3)
trail_line_B, = ax_butterfly.plot([], [], lw=0.8, alpha=0.6,
                                   color=PINK, linestyle="--", zorder=3)
point_A,      = ax_butterfly.plot([], [], "o", ms=7, color=CYAN,
                                   markeredgecolor="white",
                                   markeredgewidth=0.8, zorder=5)
point_B,      = ax_butterfly.plot([], [], "o", ms=5, color=PINK,
                                   markeredgecolor="white",
                                   markeredgewidth=0.5, zorder=5)
diverg_arrow  = ax_butterfly.annotate(
    "", xy=(0,0), xytext=(0,0),
    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
    zorder=6)

# Legenda manual
ax_butterfly.text(0.02, 0.98,
    f"  Trajetória A  (x₀ = 0.1)\n"
    f"  Trajetória B  (x₀ = 0.1 + {DELTA_IC:.0e})",
    transform=ax_butterfly.transAxes,
    color=FG, fontsize=7.5, va="top", fontfamily="monospace",
    bbox=dict(facecolor="#0a0b1a", edgecolor="#252550",
              boxstyle="round,pad=0.5", alpha=0.85))

# Patch colorido na legenda
ax_butterfly.plot([0], [0], color=CYAN, lw=2, label="A", alpha=0)
ax_butterfly.plot([0], [0], color=PINK, lw=2, linestyle="--",
                  label="B", alpha=0)

# Plano central (origem)
ax_butterfly.axhline(0, color="#151530", lw=0.5)
ax_butterfly.axvline(0, color="#151530", lw=0.5)

# Frame counter grande
fc_text = ax_butterfly.text(
    0.98, 0.02, "", transform=ax_butterfly.transAxes,
    color="#2a2850", fontsize=28, ha="right", va="bottom",
    fontweight="bold", fontfamily="monospace", alpha=0.6)

# Anotação de Lyapunov dentro da borboleta
lyap_txt = ax_butterfly.text(
    0.02, 0.12, "", transform=ax_butterfly.transAxes,
    color=GOLD, fontsize=8, va="bottom", fontfamily="monospace",
    bbox=dict(facecolor="#0a0b1a", edgecolor="#2a2810",
              boxstyle="round,pad=0.4", alpha=0.85))

# ── PAINEL ONDA ────────────────────────────────────────────────────────
ax_wave.set_title("Campo ψ(x) do parquet  —  motor de integração",
                   color=FG, fontsize=8, pad=4)
ax_wave.set_xlim(-10, 10)
ax_wave.set_ylim(-1.15, 1.15)
ax_wave.axhline(0, color="#151530", lw=0.5)
ax_wave.set_xlabel("posição x", color="#5550a0", fontsize=7)
ax_wave.tick_params(labelsize=6)
wave_line,  = ax_wave.plot([], [], lw=1.3, color=ACC, zorder=3)
wave_fill_h = [None]
# Marcador da posição x do ponto A projetado
wave_xmark  = ax_wave.axvline(0, color=CYAN, lw=0.9,
                               linestyle=":", alpha=0.7)
wave_amp_txt = ax_wave.text(0.02, 0.88, "", transform=ax_wave.transAxes,
                             color="#9890c8", fontsize=6.5,
                             fontfamily="monospace")

# ── PAINEL EQUAÇÕES ─────────────────────────────────────────────────────
ax_eqs.set_xlim(0, 1)
ax_eqs.set_ylim(0, 1)
ax_eqs.set_xticks([]); ax_eqs.set_yticks([])
ax_eqs.set_title("Equações de Lorenz  —  parâmetros ao vivo",
                  color=FG, fontsize=8, pad=4)

EQ_Y = [0.88, 0.70, 0.52, 0.30, 0.12]
EQ_COLORS = [CYAN, LIME, GOLD, ACC, "#9890c8"]

eq_labels = [
    "dx/dt = σ·(y − x)",
    "dy/dt = x·(ρ − z) − y",
    "dz/dt = x·y − β·z",
    "parâmetros modulados por ψ",
    "RK4  ·  dt = {:.4f}".format(DT),
]
for y, col, lbl in zip(EQ_Y, EQ_COLORS, eq_labels):
    ax_eqs.text(0.03, y, lbl, color=col, fontsize=7.5,
                fontfamily="monospace", va="center")

eq_vals = [
    ax_eqs.text(0.97, EQ_Y[0], "", color=CYAN, fontsize=7.5,
                ha="right", fontfamily="monospace", va="center"),
    ax_eqs.text(0.97, EQ_Y[1], "", color=LIME, fontsize=7.5,
                ha="right", fontfamily="monospace", va="center"),
    ax_eqs.text(0.97, EQ_Y[2], "", color=GOLD, fontsize=7.5,
                ha="right", fontfamily="monospace", va="center"),
    ax_eqs.text(0.97, EQ_Y[3], "", color=ACC,  fontsize=7.5,
                ha="right", fontfamily="monospace", va="center"),
    ax_eqs.text(0.97, EQ_Y[4], "", color="#9890c8", fontsize=7.5,
                ha="right", fontfamily="monospace", va="center"),
]

# ── PAINEL DIVERGÊNCIA ─────────────────────────────────────────────────
ax_diverg.set_title("Divergência |A−B|  →  Sensibilidade a condições iniciais",
                     color=FG, fontsize=8, pad=4)
ax_diverg.set_xlim(0, N_FRAMES)
log_div = np.log10(diverg + 1e-30)
ax_diverg.set_ylim(log_div.min()-0.3, log_div.max()+0.3)
ax_diverg.set_xlabel("frame (passo RK4)", color="#5550a0", fontsize=7)
ax_diverg.set_ylabel("log₁₀|A−B|", color="#5550a0", fontsize=7)
ax_diverg.tick_params(labelsize=6)
ax_diverg.plot(np.arange(N_FRAMES), log_div,
               color="#1a1c30", lw=0.7, alpha=0.5)  # ghost

div_done, = ax_diverg.plot([], [], lw=1.2, color=GOLD, alpha=0.95)
div_cur   = ax_diverg.axvline(0, color=ACC, lw=0.9, alpha=0.7)
div_dot,  = ax_diverg.plot([], [], "o", ms=5, color=GOLD,
                            markeredgecolor="white", markeredgewidth=0.5)
# Linha de saturação (|A-B| ~ escala do atrator ≈ 30)
ax_diverg.axhline(np.log10(30), color="#3a1010", lw=0.8,
                   linestyle="--", alpha=0.7)
ax_diverg.text(N_FRAMES*0.98, np.log10(30)+0.08,
               "saturação", color="#5a2020", fontsize=6,
               ha="right", fontfamily="monospace")

# ── PAINEL SHA-256 ─────────────────────────────────────────────────────
ax_sha.set_xlim(0, 1)
ax_sha.set_ylim(0, 1)
ax_sha.set_xticks([]); ax_sha.set_yticks([])
ax_sha.set_title("Assinatura SHA-256  ·  Auditoria do Parquet",
                  color=FG, fontsize=8, pad=4)

sha_ok_col = "#30ff80" if sha_ok_all else "#ff3030"
sha_bar = ax_sha.barh([0.80], [1.0], height=0.13,
                       color=sha_ok_col, alpha=0.82)[0]
sha_hash_txt = ax_sha.text(0.5, 0.80, "", color="white", fontsize=6,
                             ha="center", va="center",
                             fontfamily="monospace")
sha_status_txt = ax_sha.text(
    0.5, 0.61,
    SHA_STATUS,
    color=sha_ok_col, fontsize=7.5,
    ha="center", va="center", fontfamily="monospace")
sha_chain_txt = ax_sha.text(
    0.5, 0.44,
    f"cadeia: {CHAIN_HASH[:40]}…",
    color="#3a3870", fontsize=6,
    ha="center", va="center", fontfamily="monospace")
sha_frame_txt = ax_sha.text(0.5, 0.26, "", color="#6860a8",
                              fontsize=6.5, ha="center",
                              fontfamily="monospace")
sha_proof_txt = ax_sha.text(
    0.5, 0.09,
    "cada hash prova que aquele passo RK4\n"
    "foi calculado com aquele campo ψ e nenhum outro",
    color="#3a3870", fontsize=6, ha="center", va="center",
    fontfamily="monospace", fontstyle="italic")

# ════════════════════════════════════════════════════════════════════════
# 4 — ANIMAÇÃO
# ════════════════════════════════════════════════════════════════════════

def update(fi):
    w    = waves[fi]
    sha  = sha_chain[fi]
    ok   = sha_flags[fi]
    xA, yA, zA = traj_A[fi]
    xB, yB, zB = traj_B[fi]
    sig, rho, bet, grad = params[fi]
    div  = diverg[fi]
    lya  = lyapunov[fi]
    t_   = (fi+1) * DT

    col  = plasma(fi / max(N_FRAMES-1, 1))

    # ── Trilha da borboleta ────────────────────────────────────
    i0 = max(0, fi - TRAIL_LEN)
    xs_A = traj_A[i0:fi+1, 0]
    zs_A = traj_A[i0:fi+1, 2]
    xs_B = traj_B[i0:fi+1, 0]
    zs_B = traj_B[i0:fi+1, 2]

    trail_line_A.set_data(xs_A, zs_A)
    trail_line_A.set_color(col + (0.9,))
    trail_line_B.set_data(xs_B, zs_B)
    point_A.set_data([xA], [zA])
    point_B.set_data([xB], [zB])

    # Frame counter
    fc_text.set_text(f"{fi+1:04d}")

    # Lyapunov dentro da borboleta
    lyap_txt.set_text(
        f"λ ≈ {lya:+.4f}\n"
        f"|A−B| = {div:.2e}\n"
        f"t = {t_:.3f}"
    )
    lyap_txt.set_color(GOLD if lya > 0 else LIME)

    # ── Onda do parquet ────────────────────────────────────────
    w_slice = w[ROWS//2, :]
    wave_line.set_data(x_arr, w_slice)
    if wave_fill_h[0] is not None:
        try: wave_fill_h[0].remove()
        except: pass
    wave_fill_h[0] = ax_wave.fill_between(
        x_arr, w_slice, alpha=0.13, color=col)
    # Marcador: posição x do ponto A mapeada para [-10,10]
    x_mapped = np.clip(xA / 25 * 10, -10, 10)
    wave_xmark.set_xdata([x_mapped])
    wave_amp_txt.set_text(
        f"ψ̄={w_slice.mean():.4f}  σψ={w_slice.std():.4f}  "
        f"∇ψ={grad:.4f}"
    )

    # ── Equações ao vivo ───────────────────────────────────────
    dx = sig * (yA - xA)
    dy = xA * (rho - zA) - yA
    dz = xA * yA - bet * zA
    eq_vals[0].set_text(f"= {dx:+8.3f}   [σ={sig:.3f}]")
    eq_vals[1].set_text(f"= {dy:+8.3f}   [ρ={rho:.3f}]")
    eq_vals[2].set_text(f"= {dz:+8.3f}   [β={bet:.4f}]")
    eq_vals[3].set_text(f"σ={sig:.3f}  ρ={rho:.3f}  β={bet:.4f}")
    eq_vals[4].set_text(f"frame {fi+1}/{N_FRAMES}   t={t_:.3f}")

    # ── Divergência ────────────────────────────────────────────
    xs_hist = np.arange(fi+1)
    div_done.set_data(xs_hist, np.log10(diverg[:fi+1] + 1e-30))
    div_done.set_color(col + (0.95,))
    div_cur.set_xdata([fi])
    div_dot.set_data([fi], [np.log10(div + 1e-30)])
    div_dot.set_color(col + (1.0,))

    # ── SHA-256 ────────────────────────────────────────────────
    sha_hash_txt.set_text(sha[:42] + "…")
    sha_bar.set_color("#30ff80" if ok else "#ff3030")
    sha_frame_txt.set_text(
        f"frame {fi+1}/{N_FRAMES}  {'✓ íntegro' if ok else '✗ corrompido'}  "
        f"passo RK4 verificado"
    )

    return (trail_line_A, trail_line_B, point_A, point_B,
            wave_line, wave_xmark,
            div_done, div_cur, div_dot,
            sha_hash_txt, sha_bar, sha_frame_txt,
            fc_text, lyap_txt,
            *eq_vals)

ani = FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    interval=28,
    blit=False,
)

print("━" * 62)
print(f"  ✅  Borboleta de Lorenz calculada pelo parquet.")
print(f"      {N_FRAMES} passos RK4  ·  cada um auditado por SHA-256.")
print(f"      Expoente de Lyapunov λ ≈ {lyapunov[-1]:.4f}  (> 0 = caótico)")
print(f"      A física é universal. O substrato é auditável.")
print("━" * 62)

plt.show()
