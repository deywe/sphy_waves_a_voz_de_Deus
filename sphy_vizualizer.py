import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import hashlib
import json
import os
import sys

# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def calcular_sha256(dados: np.ndarray) -> str:
    """Recalcula o hash SHA256 de um array numpy (mesmo método do gerador)."""
    return hashlib.sha256(dados.tobytes()).hexdigest()


def validar_parquet(df: pd.DataFrame, shape_rows: int, shape_cols: int) -> tuple[bool, list]:
    """
    Recomputa o wave para cada frame e compara com o hash armazenado.
    Retorna (tudo_ok, lista_de_resultados).
    """
    x = np.linspace(-10, 10, shape_cols)
    t_window = np.linspace(0, 10, shape_rows)
    X, T_window = np.meshgrid(x, t_window)

    resultados = []
    tudo_ok = True

    print("\n  Validando integridade SHA256 de todos os frames...")
    total = len(df)

    for _, row in df.iterrows():
        frame = int(row["frame"])
        t_offset = float(row["t_offset"])
        hash_armazenado = row["sha256"]

        # Recomputa exatamente como o gerador fez
        T_animated = T_window + t_offset
        wave = np.sin(2 * np.pi * 0.3 * X - 2 * np.pi * 0.1 * T_animated) * np.exp(-0.05 * (X**2))
        hash_calculado = calcular_sha256(wave)

        ok = (hash_calculado == hash_armazenado)
        if not ok:
            tudo_ok = False
        resultados.append({"frame": frame, "ok": ok, "esperado": hash_armazenado, "calculado": hash_calculado})

        # Progresso
        if (frame + 1) % max(1, total // 10) == 0 or frame == total - 1:
            pct = (frame + 1) / total * 100
            status = "✓" if ok else "✗ FALHA"
            print(f"  [{pct:5.1f}%] Frame {frame + 1:>5}/{total} | {status} | {hash_armazenado[:16]}...")

    return tudo_ok, resultados


# ─────────────────────────────────────────────
# CARREGAR DADOS
# ─────────────────────────────────────────────

parquet_path = "sphy_frames.parquet"

if not os.path.exists(parquet_path):
    print(f"\n  ERRO: Arquivo '{parquet_path}' não encontrado.")
    print("  Execute primeiro o sphy_gerador.py para gerar os dados.\n")
    sys.exit(1)

print("=" * 60)
print("  SPHY VISUALIZER — Exotic Wave Field + SHA256 Validation  ")
print("=" * 60)
print(f"\n  Carregando: {parquet_path}")

df = pd.read_parquet(parquet_path)
total_frames = len(df)
shape_rows = int(df["shape_rows"].iloc[0])
shape_cols = int(df["shape_cols"].iloc[0])

print(f"  Frames encontrados : {total_frames}")
print(f"  Grid (linhas×cols) : {shape_rows} × {shape_cols}")

# ─────────────────────────────────────────────
# VALIDAÇÃO SHA256
# ─────────────────────────────────────────────

tudo_ok, resultados = validar_parquet(df, shape_rows, shape_cols)
falhas = [r for r in resultados if not r["ok"]]

print(f"\n{'=' * 60}")
if tudo_ok:
    print(f"  ✅ VALIDAÇÃO OK — Todos os {total_frames} frames íntegros.")
else:
    print(f"  ❌ VALIDAÇÃO FALHOU — {len(falhas)} frame(s) com hash inválido!")
    for f in falhas[:5]:
        print(f"     Frame {f['frame']}: esperado {f['esperado'][:20]}... | obtido {f['calculado'][:20]}...")
    if len(falhas) > 5:
        print(f"     ... e mais {len(falhas) - 5} falha(s).")
print(f"{'=' * 60}\n")

# ─────────────────────────────────────────────
# RECONSTRUIR WAVES PARA ANIMAÇÃO
# ─────────────────────────────────────────────

x = np.linspace(-10, 10, shape_cols)
t_window = np.linspace(0, 10, shape_rows)
X, T_window = np.meshgrid(x, t_window)

waves = []
for _, row in df.iterrows():
    t_offset = float(row["t_offset"])
    T_animated = T_window + t_offset
    wave = np.sin(2 * np.pi * 0.3 * X - 2 * np.pi * 0.1 * T_animated) * np.exp(-0.05 * (X**2))
    waves.append(wave)

# ─────────────────────────────────────────────
# ANIMAÇÃO (igual ao script original)
# ─────────────────────────────────────────────

plt.style.use("default")
fig, ax = plt.subplots(figsize=(8, 5))

# Barra de status de validação no título
status_txt = "✅ SHA256 OK" if tudo_ok else f"❌ SHA256 FALHOU ({len(falhas)} frame(s))"

init_wave = np.zeros_like(X)
quad = ax.pcolormesh(X, T_window, init_wave, shading="auto", cmap="inferno", vmin=-1, vmax=1)
plt.colorbar(quad, label="Wave Amplitude")
ax.set_xlabel("Position (x)")
ax.set_ylabel("Time History (t)")

frame_text = ax.set_title(
    f"Evolution of the Exotic Wave Field (High Speed) — {status_txt}\nFrame 0/{total_frames}",
    fontsize=10,
)

def update(i):
    quad.set_array(waves[i].ravel())
    frame_text.set_text(
        f"Evolution of the Exotic Wave Field (High Speed) — {status_txt}\n"
        f"Frame {i + 1}/{total_frames}  |  SHA256: {df['sha256'].iloc[i][:32]}..."
    )
    return quad, frame_text

ani = FuncAnimation(fig, update, frames=total_frames, interval=25, blit=True)

plt.tight_layout()
plt.show()
