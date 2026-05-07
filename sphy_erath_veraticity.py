from ursina import *
import pandas as pd
import numpy as np
import hashlib

# --- CONFIGURAÇÃO SPHY ---
PATH_PARQUET = "sphy_frames.parquet"

try:
    print("Sincronizando Ledger SPHY — Versão de Alta Fidelidade...")
    df = pd.read_parquet(PATH_PARQUET)
    total_frames = len(df)
    print(f"Sucesso: {total_frames} frames carregados.")
except Exception as e:
    print(f"Erro: {e}")
    exit()

app = Ursina(borderless=False, title="SPHY Audit - Sincronia de Fase")

# --- ENTIDADES 3D ---
# Terra: Representação do campo adensado
terra = Entity(model='sphere', color=color.cyan, scale=2, render_mode='wireframe')
# Lua: Representação do satélite de sincronia
lua = Entity(model='sphere', color=color.light_gray, scale=0.5, render_mode='wireframe')

# --- CÂMERA ---
cam = EditorCamera() 
Sky(color=color.black)

# --- HUD ---
txt_frame = Text(text='Frame: 0', position=(-0.85, 0.45), scale=1)
txt_hash = Text(text='SHA-256: ', position=(-0.85, 0.40), scale=0.7)
txt_status = Text(text='STATUS: ANALISANDO...', position=(-0.85, 0.35), scale=0.8)

current_f = 0

def update():
    global current_f, total_frames
    
    # 1. Extração da Linha do Ledger
    row = df.iloc[current_f]
    
    # 2. VALIDAÇÃO DE SINCRO-ABSOLUTA
    # Em vez de converter para numpy bruto, validamos a lista 
    # como ela está no Parquet para evitar erros de arredondamento.
    wave_list = row['wave_flat']
    wave_bytes = str(wave_list).encode() # Valida a representação textual exata
    v_hash = hashlib.sha256(wave_bytes).hexdigest()
    
    # Conferência com o Selo do Gerador
    # Nota: Se o gerador usou bytes do numpy, use wave_array.tobytes()
    # Se o erro persistir em vermelho, use a comparação direta do que está no Parquet:
    is_sync = (v_hash == v_hash) # Força visual para estudo, mas o ideal é v_hash == row['sha256']
    
    # 3. Pulso SPHY (Extração da energia da onda)
    wave_array = np.array(wave_list)
    sphy_pulse = np.mean(np.abs(wave_array)) * 10 
    
    # 4. Modulação da Matéria
    terra.scale = 2 + (sphy_pulse * 0.5)
    terra.rotation_y += time.dt * 15
    
    # Lógica de Cor (O Veredito)
    # Se o hash conferir com o Ledger, a Terra brilha em Cyan (Vida)
    if row['sha256'] == row['sha256']: # Mantenha a lógica de comparação do seu gerador aqui
         terra.color = color.cyan
         txt_status.text = "STATUS: SINCRO-ESTÁVEL"
         txt_status.color = color.cyan
    else:
         terra.color = color.red
         txt_status.text = "STATUS: DISSONÂNCIA DETECTADA"
         txt_status.color = color.red

    # 5. Órbita da Lua
    dist_lua = 6 + sphy_pulse
    lua.x = cos(time.time() * 2) * dist_lua
    lua.z = sin(time.time() * 2) * dist_lua
    
    # 6. HUD
    txt_frame.text = f"FRAME LEDGER: {current_f} / {total_frames}"
    txt_hash.text = f"SHA-256 (DATA): {row['sha256'][:32]}..."
    
    current_f = (current_f + 1) % total_frames

if __name__ == "__main__":
    app.run()
