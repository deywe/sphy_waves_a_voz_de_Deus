from ursina import *
import pandas as pd
import numpy as np
import hashlib

# --- CONFIGURAÇÃO SPHY ---
PATH_PARQUET = "sphy_frames.parquet"

try:
    print("Sincronizando Ledger via Ursina Engine...")
    df = pd.read_parquet(PATH_PARQUET)
    total_frames = len(df)
    print(f"Sucesso: {total_frames} frames carregados.")
except Exception as e:
    print(f"Erro ao carregar Parquet: {e}")
    exit()

app = Ursina(borderless=False, title="SPHY Audit Visualizer - Navegação Estabilizada")

# --- ENTIDADES 3D ---
terra = Entity(model='sphere', color=color.cyan, scale=2, render_mode='wireframe')
lua = Entity(model='sphere', color=color.light_gray, scale=0.5, render_mode='wireframe')

# --- CÂMERA E CONTROLES ---
# EditorCamera já possui: 
# - Botão Direito: Rotacionar
# - Scroll Mouse: Zoom
# - Botão do Meio: Mover (Pan)
cam = EditorCamera() 

Sky(color=color.black)

# --- HUD ---
txt_frame = Text(text='Frame: 0', position=(-0.85, 0.45), scale=1)
txt_hash = Text(text='SHA-256: ', position=(-0.85, 0.40), scale=0.7, color=color.lime)
txt_status = Text(text='STATUS: SINCRO-ESTÁVEL', position=(-0.85, 0.35), scale=0.8)

current_f = 0

def update():
    global current_f, total_frames
    
    # Leitura do Ledger
    row = df.iloc[current_f]
    wave_flat = np.array(row['wave_flat'], dtype=np.float32)
    
    # Validação SHA-256
    v_hash = hashlib.sha256(wave_flat.tobytes()).hexdigest()
    is_sync = (v_hash == row['sha256'])
    
    # Pulso SPHY
    sphy_pulse = np.mean(np.abs(wave_flat)) * 8 
    
    # Geometria
    terra.scale = 2 + sphy_pulse
    terra.rotation_y += time.dt * 15
    
    if is_sync:
        terra.color = color.cyan
        txt_status.text = "STATUS: SINCRO-ESTÁVEL"
        txt_status.color = color.lime
    else:
        terra.color = color.red
        txt_status.text = "STATUS: DISSONÂNCIA"
        txt_status.color = color.red

    # Órbita da Lua
    dist_lua = 6 + sphy_pulse
    lua.x = cos(time.time() * 2.5) * dist_lua
    lua.z = sin(time.time() * 2.5) * dist_lua
    
    # HUD
    txt_frame.text = f"FRAME LEDGER: {current_f} / {total_frames}"
    txt_hash.text = f"SHA-256: {v_hash[:32]}..."
    
    current_f = (current_f + 1) % total_frames

if __name__ == "__main__":
    app.run()
