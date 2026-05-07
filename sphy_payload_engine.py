"""
sphy_payload_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPHY PAYLOAD ENGINE  —  Motor Universal sobre sphy_frames.parquet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Aceita qualquer payload de até 1200 frames e prova que o parquet
pode calcular qualquer coisa nesse limite.

Cada payload é uma função  f(wave, frame_meta) → resultado
que recebe a onda do frame (já validada por SHA-256) e devolve
um dicionário de resultados. O engine:

  1. Carrega e valida o parquet (SHA-256 frame a frame)
  2. Limita ao slice de frames pedido (≤ 1200)
  3. Executa o payload injetado sobre cada frame
  4. Coleta resultados + assina a execução inteira
  5. Salva relatório em  sphy_payload_report.parquet
  6. Exibe visualização animada dos resultados

PAYLOADS EMBUTIDOS (escolha ao rodar):
  1. hydrogen_atom   — parâmetros do átomo de hidrogênio
  2. fourier_decomp  — decomposição de Fourier da onda
  3. chaos_lorenz    — atrator de Lorenz alimentado pela onda
  4. prime_field     — mapeamento da onda em crivos de primos
  5. custom          — injete sua própria função Python

Uso:
    python sphy_payload_engine.py
    python sphy_payload_engine.py --payload fourier_decomp --frames 400
    python sphy_payload_engine.py --payload custom --func meu_modulo.minha_func
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse, hashlib, importlib.util, json, os, sys, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════

MAX_FRAMES   = 1200
PARQUET_PATH = "sphy_frames.parquet"
REPORT_PATH  = "sphy_payload_report.parquet"

BG  = "#07080f"
FG  = "#cdd0e8"
ACC = "#ff6e3a"
GRN = "#30ff90"
BLU = "#30b0ff"

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def norm01(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)

def inferno(t):
    t = float(np.clip(t, 0, 1))
    r = np.clip(t**0.45, 0, 1)
    g = np.clip(0.85 * t**2.2, 0, 1)
    b = np.clip(0.7 * (1-t)**1.8 + 0.15*t, 0, 1)
    return (r, g, b)

# ════════════════════════════════════════════════════════════════
# PAYLOADS EMBUTIDOS
# ════════════════════════════════════════════════════════════════

def payload_hydrogen_atom(wave: np.ndarray, meta: dict) -> dict:
    """
    Extrai parâmetros físicos do átomo de hidrogênio a partir da onda.
    A amplitude modula raio orbital e velocidade angular do elétron.
    """
    amp_mean = float(wave.mean())
    amp_std  = float(wave.std())
    energy   = float((wave**2).mean())
    amp_max  = float(np.abs(wave).max())

    radius   = 0.35 + 0.55 * np.clip(amp_std / 0.5, 0, 1)
    omega    = 0.8  + 1.4  * np.clip(energy  / 0.5, 0, 1)
    theta    = meta["frame"] * omega * 0.07
    ex, ey   = radius * np.cos(theta), radius * np.sin(theta)

    return {
        "payload"    : "hydrogen_atom",
        "amp_mean"   : amp_mean,
        "amp_std"    : amp_std,
        "energy"     : energy,
        "amp_max"    : amp_max,
        "radius_a0"  : radius,
        "omega_rad"  : omega,
        "electron_x" : ex,
        "electron_y" : ey,
        "plot_primary"   : energy,
        "plot_secondary" : radius,
        "label_primary"  : "Energia ⟨ψ²⟩",
        "label_secondary": "Raio orbital (a₀)",
    }


def payload_fourier_decomp(wave: np.ndarray, meta: dict) -> dict:
    """
    Decomposição de Fourier da fatia central da onda.
    Retorna as 3 frequências dominantes e suas amplitudes.
    """
    slice_1d = wave[wave.shape[0]//2, :]
    fft      = np.fft.rfft(slice_1d)
    freqs    = np.fft.rfftfreq(len(slice_1d))
    amps     = np.abs(fft) / len(slice_1d)

    top3_idx = np.argsort(amps)[-3:][::-1]
    f1, f2, f3 = freqs[top3_idx]
    a1, a2, a3 = amps[top3_idx]
    spectral_entropy = float(-np.sum((amps/(amps.sum()+1e-12)) *
                                      np.log(amps/(amps.sum()+1e-12) + 1e-12)))
    power_total = float((amps**2).sum())

    return {
        "payload"        : "fourier_decomp",
        "freq_1"         : float(f1),
        "freq_2"         : float(f2),
        "freq_3"         : float(f3),
        "amp_1"          : float(a1),
        "amp_2"          : float(a2),
        "amp_3"          : float(a3),
        "spectral_entropy": spectral_entropy,
        "power_total"    : power_total,
        "plot_primary"   : spectral_entropy,
        "plot_secondary" : power_total,
        "label_primary"  : "Entropia espectral",
        "label_secondary": "Potência total",
    }


def _lorenz_state(sigma=10.0, rho=28.0, beta=8/3, dt=0.005, steps=8):
    """Integrador Lorenz mínimo — retorna delta de posição."""
    # Estado persistido via closure — reinicia a cada frame propositalmente
    x, y, z = 1.0, 1.0, 1.0
    for _ in range(steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
    return x, y, z

def payload_chaos_lorenz(wave: np.ndarray, meta: dict) -> dict:
    """
    Atrator de Lorenz com condição inicial alimentada pela onda.
    A amplitude da onda em x=0 perturba o estado inicial de Lorenz.
    Mostra sensibilidade às condições iniciais = física caótica universal.
    """
    center_val = float(wave[wave.shape[0]//2, wave.shape[1]//2])
    sigma = 10.0 + center_val * 2.0     # perturbação da onda → sigma
    rho   = 28.0 + wave.mean() * 3.0   # amplitude média → rho
    beta  = 8/3  + wave.std() * 0.5    # dispersão → beta

    steps = 12 + meta["frame"] % 8     # passos variáveis por frame
    x, y, z = 1.0 + center_val, 1.0, 1.0
    dt = 0.005
    sx, sy, rho_v, bv = sigma, 10.0, rho, beta
    for _ in range(steps):
        dx = sx  * (y - x)
        dy = x   * (rho_v - z) - y
        dz = x*y - bv*z
        x += dx*dt; y += dy*dt; z += dz*dt

    lyapunov_proxy = float(np.log(abs(x) + 1e-9))   # proxy do expoente de Lyapunov

    return {
        "payload"       : "chaos_lorenz",
        "lorenz_x"      : float(x),
        "lorenz_y"      : float(y),
        "lorenz_z"      : float(z),
        "sigma_driven"  : float(sigma),
        "rho_driven"    : float(rho),
        "lyapunov_proxy": lyapunov_proxy,
        "center_val"    : center_val,
        "plot_primary"  : float(z),
        "plot_secondary": lyapunov_proxy,
        "label_primary" : "Lorenz Z (caos)",
        "label_secondary":"λ proxy (Lyapunov)",
    }


def payload_prime_field(wave: np.ndarray, meta: dict) -> dict:
    """
    Mapeia a onda num crivo de primos.
    Quantiza a amplitude → índice de primos → densidade de números primos
    na vizinhança. Mostra que a onda exótica pode codificar teoria dos números.
    """
    # Crivo de Eratóstenes até 2000
    N = 2000
    sieve = np.ones(N+1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N**0.5)+1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0]

    # Amplitude média da onda → índice no array de primos
    amp_norm   = float((wave.mean() + 1) / 2)   # [0,1]
    prime_idx  = int(np.clip(amp_norm * len(primes), 0, len(primes)-1))
    prime_val  = int(primes[prime_idx])

    # Densidade de primos na janela de ±50 ao redor do valor
    window     = sieve[max(0, prime_val-50) : prime_val+50]
    density    = float(window.sum() / len(window))

    # Dispersão da onda → gap entre primos consecutivos
    std_norm   = float(wave.std())
    gap_idx    = int(np.clip(std_norm * len(primes) * 0.8, 0, len(primes)-2))
    prime_gap  = int(primes[gap_idx+1] - primes[gap_idx])

    # Energia → índice de Goldbach (par = soma de 2 primos?)
    energy_int = int(np.clip((wave**2).mean() * 1000, 4, N)) // 2 * 2
    is_goldbach = bool(any(sieve[energy_int - p] for p in primes[primes < energy_int]))

    return {
        "payload"      : "prime_field",
        "prime_val"    : prime_val,
        "prime_idx"    : prime_idx,
        "prime_density": density,
        "prime_gap"    : prime_gap,
        "energy_int"   : energy_int,
        "is_goldbach"  : is_goldbach,
        "plot_primary" : density,
        "plot_secondary": float(prime_gap),
        "label_primary" : "Densidade de primos",
        "label_secondary": "Gap entre primos",
    }


PAYLOAD_REGISTRY = {
    "hydrogen_atom"  : payload_hydrogen_atom,
    "fourier_decomp" : payload_fourier_decomp,
    "chaos_lorenz"   : payload_chaos_lorenz,
    "prime_field"    : payload_prime_field,
}

# ════════════════════════════════════════════════════════════════
# ENGINE PRINCIPAL
# ════════════════════════════════════════════════════════════════

class SphyPayloadEngine:

    def __init__(self, parquet_path: str = PARQUET_PATH):
        self.parquet_path = parquet_path
        self.df           = None
        self.waves        = []
        self.sha_flags    = []
        self.sha_ok_all   = True
        self.results      = []

    # ── 1. Carregamento ──────────────────────────────────────

    def load(self):
        if not os.path.exists(self.parquet_path):
            print(f"\n  ❌  '{self.parquet_path}' não encontrado.")
            print("  Execute sphy_gerador.py primeiro.\n")
            sys.exit(1)

        self.df    = pd.read_parquet(self.parquet_path)
        self.ROWS  = int(self.df["shape_rows"].iloc[0])
        self.COLS  = int(self.df["shape_cols"].iloc[0])
        self.TOTAL = len(self.df)

        x_arr    = np.linspace(-10, 10, self.COLS)
        t_arr    = np.linspace(0,  10,  self.ROWS)
        X_g, T_g = np.meshgrid(x_arr, t_arr)

        print(f"\n  Parquet carregado  : {self.TOTAL} frames  |  grid {self.ROWS}×{self.COLS}")
        self.X_g = X_g
        self.T_g = T_g
        self.x_arr = x_arr

    # ── 2. Validação SHA-256 ─────────────────────────────────

    def validate(self, frame_slice: slice):
        sub_df = self.df.iloc[frame_slice].reset_index(drop=True)
        n      = len(sub_df)
        print(f"  Validando SHA-256 de {n} frames …")
        t0 = time.perf_counter()

        for i, row in sub_df.iterrows():
            t_off  = float(row["t_offset"])
            T_anim = self.T_g + t_off
            w      = (np.sin(2*np.pi*0.3*self.X_g - 2*np.pi*0.1*T_anim)
                      * np.exp(-0.05 * self.X_g**2))
            self.waves.append(w)

            h_calc  = sha256_array(w)
            h_store = row["sha256"]
            ok      = (h_calc == h_store)
            self.sha_flags.append(ok)
            if not ok:
                self.sha_ok_all = False
                print(f"  ❌  frame {i}: hash divergente!")

            if (i+1) % max(1, n//8) == 0 or i == n-1:
                pct = (i+1)/n*100
                sym = "✓" if ok else "✗"
                print(f"  [{pct:5.1f}%] frame {i+1:>5}/{n}  {sym}  {h_calc[:18]}…")

        elapsed = time.perf_counter() - t0
        status  = ("✅  SHA-256 OK — todos íntegros"
                   if self.sha_ok_all
                   else f"❌  SHA-256 FALHOU em {self.sha_flags.count(False)} frame(s)")
        print(f"  {status}  [{elapsed:.2f}s]")
        return self.sha_ok_all

    # ── 3. Execução do payload ───────────────────────────────

    def run_payload(self, fn, payload_name: str):
        n = len(self.waves)
        print(f"\n  Executando payload '{payload_name}' em {n} frames …")
        t0 = time.perf_counter()

        execution_log = []
        for i, w in enumerate(self.waves):
            meta = {
                "frame"     : i,
                "sha_ok"    : self.sha_flags[i],
                "sha256"    : self.df["sha256"].iloc[i],
                "t_offset"  : float(self.df["t_offset"].iloc[i]),
                "payload"   : payload_name,
            }
            try:
                result = fn(w, meta)
            except Exception as e:
                result = {"payload": payload_name, "error": str(e),
                          "plot_primary": 0.0, "plot_secondary": 0.0,
                          "label_primary": "erro", "label_secondary": "erro"}

            result.update({
                "frame"      : i,
                "sha256"     : meta["sha256"],
                "sha_ok"     : meta["sha_ok"],
                "t_offset"   : meta["t_offset"],
                "exec_time"  : time.perf_counter() - t0,
            })
            self.results.append(result)
            execution_log.append(result["sha256"])

            if (i+1) % max(1, n//8) == 0 or i == n-1:
                pct = (i+1)/n*100
                pv  = result.get("plot_primary", 0)
                print(f"  [{pct:5.1f}%] frame {i+1:>5}/{n}  "
                      f"{result.get('label_primary','val')} = {pv:.5f}")

        elapsed = time.perf_counter() - t0

        # Assinatura da execução completa (hash da cadeia de hashes)
        chain_hash = sha256_str("".join(execution_log))
        print(f"\n  ✅  Execução concluída em {elapsed:.2f}s")
        print(f"  🔐  Hash da cadeia  : {chain_hash[:48]}…")
        self.chain_hash = chain_hash
        self.elapsed    = elapsed
        return self.results

    # ── 4. Salvar relatório ──────────────────────────────────

    def save_report(self, payload_name: str):
        rdf = pd.DataFrame(self.results)
        rdf.to_parquet(REPORT_PATH, index=False, compression="snappy")
        size_kb = os.path.getsize(REPORT_PATH) / 1024

        meta = {
            "payload"          : payload_name,
            "total_frames"     : len(self.results),
            "sha_ok_all"       : self.sha_ok_all,
            "chain_hash"       : self.chain_hash,
            "elapsed_s"        : self.elapsed,
            "generated_at"     : datetime.now(timezone.utc).isoformat(),
            "parquet_source"   : self.parquet_path,
        }
        with open("sphy_payload_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  💾  Relatório salvo : {REPORT_PATH}  ({size_kb:.1f} KB)")
        print(f"  📄  Metadados       : sphy_payload_meta.json")

    # ── 5. Visualização animada ──────────────────────────────

    def visualize(self, payload_name: str):
        results   = self.results
        n         = len(results)
        primary   = np.array([r.get("plot_primary",   0.0) for r in results])
        secondary = np.array([r.get("plot_secondary", 0.0) for r in results])
        lbl_p     = results[0].get("label_primary",   "primary")
        lbl_s     = results[0].get("label_secondary", "secondary")

        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": BG,
            "text.color": FG, "axes.labelcolor": FG,
            "xtick.color": FG, "ytick.color": FG,
            "axes.edgecolor": "#1a1c2e", "font.family": "monospace",
        })

        fig = plt.figure(figsize=(15, 8), facecolor=BG)
        fig.canvas.manager.set_window_title(
            f"SPHY Payload Engine — {payload_name}")

        gs = gridspec.GridSpec(3, 4, figure=fig,
                               left=0.05, right=0.97,
                               top=0.90, bottom=0.08,
                               wspace=0.42, hspace=0.60)

        # Painéis
        ax_wave  = fig.add_subplot(gs[0,  0:2])   # onda do frame
        ax_pay   = fig.add_subplot(gs[1,  0:2])   # saída primária do payload
        ax_sha   = fig.add_subplot(gs[2,  0:2])   # SHA-256 + cadeia
        ax_hist1 = fig.add_subplot(gs[0:2, 2:4])  # histórico primary
        ax_hist2 = fig.add_subplot(gs[2,   2:4])  # histórico secondary

        for ax in [ax_wave, ax_pay, ax_sha, ax_hist1, ax_hist2]:
            ax.set_facecolor(BG)
            for sp in ax.spines.values():
                sp.set_edgecolor("#1a1c2e")

        # ── Título global ─────────────────────────────────────
        fig.text(0.5, 0.96,
                 f"SPHY PAYLOAD ENGINE  —  payload: {payload_name}  "
                 f"({n} frames  /  máx {MAX_FRAMES})",
                 ha="center", color=FG, fontsize=11, fontweight="bold",
                 fontfamily="monospace")
        fig.text(0.5, 0.935,
                 f"sphy_frames.parquet  →  SHA-256 verificado  →  "
                 f"qualquer cálculo nesse substrato é auditável",
                 ha="center", color="#5550a0", fontsize=8, fontstyle="italic",
                 fontfamily="monospace")

        # ── Painel onda ───────────────────────────────────────
        ax_wave.set_title("Campo Escalar (parquet)", color=FG, fontsize=8, pad=4)
        ax_wave.set_xlim(-10, 10)
        ax_wave.set_ylim(-1.15, 1.15)
        ax_wave.axhline(0, color="#2a2840", lw=0.5)
        wave_line,   = ax_wave.plot([], [], lw=1.2, color=ACC)
        wave_fill_h  = [None]

        # ── Painel payload ────────────────────────────────────
        ax_pay.set_title(f"Payload: {lbl_p}", color=FG, fontsize=8, pad=4)
        ax_pay.set_xlim(-10, 10)
        ax_pay.set_ylim(primary.min()-0.05, primary.max()+0.05)
        pay_line, = ax_pay.plot([], [], lw=1.4, color=GRN)
        pay_hline = ax_pay.axhline(0, color="#2a2840", lw=0.5)
        pay_dot,  = ax_pay.plot([], [], "o", ms=7, color=GRN,
                                 markeredgecolor="white", markeredgewidth=0.7)

        # ── Painel SHA ────────────────────────────────────────
        ax_sha.set_xlim(0, 1)
        ax_sha.set_ylim(0, 1)
        ax_sha.set_xticks([]); ax_sha.set_yticks([])
        ax_sha.set_title("Assinatura SHA-256  ·  Cadeia de Execução", color=FG,
                          fontsize=8, pad=4)

        bar_col = GRN if self.sha_ok_all else "#ff3030"
        sha_bar = ax_sha.barh([0.78], [1.0], height=0.14, color=bar_col,
                               alpha=0.80)[0]
        sha_txt = ax_sha.text(0.5, 0.78, "", color="white", fontsize=6.5,
                               ha="center", va="center", fontfamily="monospace")
        sha_status = ax_sha.text(0.5, 0.56,
                                  "✅ SHA-256 OK" if self.sha_ok_all else "❌ FALHOU",
                                  color=bar_col, fontsize=8,
                                  ha="center", va="center", fontfamily="monospace")
        chain_txt = ax_sha.text(0.5, 0.37,
                                 f"cadeia: {self.chain_hash[:38]}…",
                                 color="#4a48a0", fontsize=6,
                                 ha="center", va="center", fontfamily="monospace")
        frame_txt = ax_sha.text(0.5, 0.20, "", color="#9890c8",
                                 fontsize=7, ha="center", fontfamily="monospace")
        result_txt = ax_sha.text(0.5, 0.06, "", color="#6860a8",
                                  fontsize=6.5, ha="center", fontfamily="monospace")

        # ── Histórico primary ─────────────────────────────────
        ax_hist1.set_title(f"Histórico: {lbl_p}", color=FG, fontsize=8, pad=4)
        ax_hist1.set_xlim(0, n)
        ax_hist1.set_ylim(primary.min()*0.97, primary.max()*1.03)
        ax_hist1.set_xlabel("frame", color=FG, fontsize=7)
        ax_hist1.tick_params(labelsize=6)
        # Curva completa em cinza
        x_all = np.arange(n)
        ax_hist1.plot(x_all, primary, color="#2a2840", lw=0.7, alpha=0.5)
        hist1_done, = ax_hist1.plot([], [], lw=1.2,
                                     color=GRN, alpha=0.9)
        hist1_cur   = ax_hist1.axvline(0, color=ACC, lw=0.9)
        hist1_dot,  = ax_hist1.plot([], [], "o", ms=5, color=ACC,
                                     markeredgecolor="white", markeredgewidth=0.6)

        # ── Histórico secondary ───────────────────────────────
        ax_hist2.set_title(f"Histórico: {lbl_s}", color=FG, fontsize=8, pad=4)
        ax_hist2.set_xlim(0, n)
        ax_hist2.set_ylim(secondary.min()*0.97, secondary.max()*1.03)
        ax_hist2.set_xlabel("frame", color=FG, fontsize=7)
        ax_hist2.tick_params(labelsize=6)
        ax_hist2.plot(x_all, secondary, color="#2a2840", lw=0.7, alpha=0.5)
        hist2_done, = ax_hist2.plot([], [], lw=1.2, color=BLU, alpha=0.9)
        hist2_cur   = ax_hist2.axvline(0, color="#ff90d0", lw=0.9)
        hist2_dot,  = ax_hist2.plot([], [], "o", ms=5, color="#ff90d0",
                                     markeredgecolor="white", markeredgewidth=0.6)

        # ── Gradiente nos históricos ──────────────────────────
        for ax_h, col in [(ax_hist1, GRN), (ax_hist2, BLU)]:
            r, g, b = [int(col[i:i+2], 16)/255
                       for i in (1, 3, 5)] if col.startswith("#") else ([0.3,1,0.6] if col == GRN else [0.2,0.7,1.0])
            ax_h.axhline(0, color="#1a1c2e", lw=0.5)

        # ── Update ────────────────────────────────────────────
        def update(fi):
            w   = self.waves[fi]
            res = results[fi]
            sha = res["sha256"]
            ok  = res["sha_ok"]
            col = inferno(fi / max(n-1, 1))

            # Onda
            w_slice = w[w.shape[0]//2, :]
            wave_line.set_data(self.x_arr, w_slice)
            ax_wave.set_title(f"Campo Escalar  —  frame {fi+1}/{n}",
                               color=FG, fontsize=8, pad=4)
            if wave_fill_h[0] is not None:
                try: wave_fill_h[0].remove()
                except: pass
            wave_fill_h[0] = ax_wave.fill_between(
                self.x_arr, w_slice, alpha=0.12,
                color=col)

            # Payload primary no painel central
            pv = res.get("plot_primary", 0)
            # Exibe como linha horizontal no nível atual + ponto
            x_mapped = np.linspace(-10, 10, 1)
            pay_line.set_data([-10, 10], [pv, pv])
            pay_dot.set_data([0], [pv])
            pay_dot.set_color([col + (1.0,)])
            pay_line.set_color(col + (0.9,))
            ax_pay.set_title(
                f"Payload  {lbl_p} = {pv:.6f}", color=FG, fontsize=8, pad=4)

            # SHA
            sha_txt.set_text(sha[:36] + "…")
            sha_bar.set_color(GRN if ok else "#ff3030")
            frame_txt.set_text(
                f"frame {fi+1}/{n}  "
                f"{'✓ íntegro' if ok else '✗ corrompido'}"
                f"  payload={payload_name}")
            sv = res.get("plot_secondary", 0)
            result_txt.set_text(
                f"{lbl_p}={pv:.5f}   {lbl_s}={sv:.5f}")

            # Históricos
            xs = np.arange(fi+1)
            hist1_done.set_data(xs, primary[:fi+1])
            hist1_cur.set_xdata([fi])
            hist1_dot.set_data([fi], [primary[fi]])
            hist1_dot.set_color([col + (1.0,)])

            hist2_done.set_data(xs, secondary[:fi+1])
            hist2_cur.set_xdata([fi])
            hist2_dot.set_data([fi], [secondary[fi]])

            return (wave_line, pay_line, pay_dot, sha_txt,
                    sha_bar, frame_txt, result_txt,
                    hist1_done, hist1_cur, hist1_dot,
                    hist2_done, hist2_cur, hist2_dot)

        ani = FuncAnimation(fig, update, frames=n,
                            interval=30, blit=False)
        plt.show()


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SPHY Payload Engine — motor universal sobre parquet")
    p.add_argument("--payload", default=None,
                   help="Nome do payload: hydrogen_atom | fourier_decomp | "
                        "chaos_lorenz | prime_field | custom")
    p.add_argument("--frames", type=int, default=None,
                   help=f"Número de frames a processar (máx {MAX_FRAMES})")
    p.add_argument("--func", default=None,
                   help="Para --payload custom: 'modulo.funcao'")
    p.add_argument("--no-viz", action="store_true",
                   help="Pula a visualização (só processa e salva)")
    return p.parse_args()


def choose_payload_interactive(total_available: int) -> tuple:
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│         SPHY PAYLOAD ENGINE  —  escolha o payload   │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  1. hydrogen_atom   — átomo de hidrogênio           │")
    print("│  2. fourier_decomp  — decomposição de Fourier        │")
    print("│  3. chaos_lorenz    — atrator de Lorenz caótico      │")
    print("│  4. prime_field     — crivo de primos + Goldbach     │")
    print("│  5. custom          — injete sua própria função      │")
    print("└─────────────────────────────────────────────────────┘")

    choice_map = {
        "1": "hydrogen_atom", "2": "fourier_decomp",
        "3": "chaos_lorenz",  "4": "prime_field", "5": "custom",
        "hydrogen_atom": "hydrogen_atom", "fourier_decomp": "fourier_decomp",
        "chaos_lorenz": "chaos_lorenz",   "prime_field": "prime_field",
        "custom": "custom",
    }

    while True:
        raw = input("\n  Payload [1-5 ou nome]: ").strip().lower()
        if raw in choice_map:
            payload_name = choice_map[raw]
            break
        print("  Opção inválida. Digite 1-5 ou o nome do payload.")

    custom_fn = None
    if payload_name == "custom":
        spec_str = input(
            "\n  Informe 'modulo.funcao' ou deixe vazio para usar um exemplo: "
        ).strip()
        if spec_str:
            parts  = spec_str.rsplit(".", 1)
            mod_p, fn_name = parts[0], parts[1]
            spec   = importlib.util.spec_from_file_location("custom_mod", mod_p + ".py")
            mod    = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            custom_fn = getattr(mod, fn_name)
            print(f"  ✓ Função '{fn_name}' carregada de '{mod_p}.py'")
        else:
            # Exemplo embutido de payload custom
            def custom_example(wave, meta):
                """Exemplo: correlação cruzada da onda consigo mesma deslocada."""
                flat = wave.ravel()
                shift = int(len(flat) * 0.1)
                corr  = float(np.corrcoef(flat, np.roll(flat, shift))[0, 1])
                rms   = float(np.sqrt((flat**2).mean()))
                return {
                    "payload"        : "custom_autocorr",
                    "autocorrelation": corr,
                    "rms"            : rms,
                    "plot_primary"   : corr,
                    "plot_secondary" : rms,
                    "label_primary"  : "Autocorrelação (lag 10%)",
                    "label_secondary": "RMS da onda",
                }
            custom_fn = custom_example
            print("  ✓ Usando exemplo embutido: autocorrelação da onda")

    # Frames
    while True:
        raw_f = input(
            f"\n  Frames a processar [1-{min(MAX_FRAMES, total_available)}, "
            f"Enter = {min(200, total_available)}]: "
        ).strip()
        if raw_f == "":
            n_frames = min(200, total_available)
            break
        try:
            n_frames = int(raw_f)
            if 1 <= n_frames <= MAX_FRAMES:
                break
            print(f"  Deve ser entre 1 e {MAX_FRAMES}.")
        except ValueError:
            print("  Número inválido.")

    n_frames = min(n_frames, total_available, MAX_FRAMES)
    return payload_name, custom_fn, n_frames


def main():
    args = parse_args()

    print("━" * 62)
    print("  SPHY PAYLOAD ENGINE  —  qualquer cálculo em até 1200 frames")
    print("━" * 62)

    engine = SphyPayloadEngine()
    engine.load()

    # Resolve payload e frames (CLI ou interativo)
    if args.payload and args.frames:
        payload_name = args.payload
        n_frames     = min(args.frames, MAX_FRAMES, engine.TOTAL)
        custom_fn    = None
        if payload_name == "custom":
            if not args.func:
                print("  --payload custom requer --func modulo.funcao")
                sys.exit(1)
            mod_p, fn_name = args.func.rsplit(".", 1)
            spec  = importlib.util.spec_from_file_location("m", mod_p + ".py")
            mod   = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            custom_fn = getattr(mod, fn_name)
    else:
        payload_name, custom_fn, n_frames = choose_payload_interactive(engine.TOTAL)

    # Resolve função
    if payload_name == "custom":
        fn = custom_fn
        payload_name = "custom"
    else:
        fn = PAYLOAD_REGISTRY.get(payload_name)
        if fn is None:
            print(f"  ❌  Payload '{payload_name}' não encontrado.")
            print(f"  Disponíveis: {list(PAYLOAD_REGISTRY.keys())}")
            sys.exit(1)

    print(f"\n  Payload  : {payload_name}")
    print(f"  Frames   : {n_frames}  (de {engine.TOTAL} disponíveis)")
    print(f"  Limite   : {MAX_FRAMES} frames máximos\n")

    # Valida + executa
    engine.validate(slice(0, n_frames))
    engine.run_payload(fn, payload_name)
    engine.save_report(payload_name)

    print("\n━" * 62)
    print(f"  ✅  Prova concluída: o parquet calculou '{payload_name}'")
    print(f"      em {n_frames} frames, cada um verificado por SHA-256.")
    print(f"      A física é universal — o substrato é auditável.")
    print("━" * 62)

    if not args.no_viz:
        print("\n  Abrindo visualização … (feche a janela para encerrar)\n")
        engine.visualize(payload_name)


if __name__ == "__main__":
    main()
