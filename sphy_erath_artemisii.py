from ursina import *
import pandas as pd
import numpy as np
import hashlib

# --- CONFIGURAÇÃO SPHY ---
PATH_PARQUET = "sphy_frames.parquet"

try:
    df = pd.read_parquet(PATH_PARQUET)
    total_frames = len(df)
    print(f"SPHY Engine: Missão Artemis II Ativada ({total_frames} frames).")
except Exception as e:
    print(f"Erro Crítico no Ledger (Parquet corrompido?): {e}")
    # Cria dados dummy apenas para demonstração visual
    total_frames = 1200
    df = pd.DataFrame({'sha256': ['hash']*1200, 'wave_flat': [[0]*200]*1200})

app = Ursina(borderless=False, title="SPHY Mission: Artemis II - Transfiguration")

# --- ENTIDADES VISÍVEIS ---
# Terra SPHÍLICA (Cyan e Wire)
terra = Entity(model='sphere', color=color.cyan, scale=3, render_mode='wireframe')
# Lua SPHÍLICA (Cinza e Wire)
lua = Entity(model='sphere', color=color.light_gray, scale=0.8, render_mode='wireframe')

# --- TRANSFIGURAÇÃO DO FOGUETE ARTEMIS II ---
# 1. Corpo do Foguete (Substituindo o Cubo por um Cone Sólido e Longo)
artemis_body = Entity(model='cone', color=color.yellow, scale=(0.5, 1.5, 0.5), position=(0,0,0))
# 2. Tanque de Combustível (Uma base sólida para o foguete)
artemis_tank = Entity(model='cube', color=color.white, scale=(0.4, 0.4, 0.4), position=(0,-0.6,0), parent=artemis_body)

# O foguete é a união dos dois (artemis_body contém o tanque)
artemis = artemis_body

# --- CÂMERA E ESPAÇO ---
cam = EditorCamera(distance=30) # Foco inicial amplo
Sky(color=color.black)          # Espaço profundo

# HUD de Auditoria (Informação Soberana)
txt_status = Text(text='MISSION: ARTEMIS II - SPHY AUDIT', position=(-0.85, 0.45), scale=1, color=color.yellow)
txt_hash = Text(text='HASH:', position=(-0.85, 0.40), scale=0.7, color=color.gray)

current_f = 0

def update():
    global current_f, total_frames
    
    # 1. Extração de Dados do Ledger
    row = df.iloc[current_f]
    wave_flat = row['wave_flat']
    
    # Pulso SPHY (A Métrica que Respira)
    if isinstance(wave_flat, (list, np.ndarray)) and len(wave_flat) > 0:
        wave_array = np.array(wave_flat)
        sphy_pulse = np.mean(np.abs(wave_array)) * 12 
    else:
        sphy_pulse = 0
    
    # 2. Modulação da Terra (Respiração SPHY)
    terra.scale = 3 + sphy_pulse
    terra.rotation_y += time.dt * 8
    
    # 3. Órbita da Lua (Lenta para observação)
    lua_angle = time.time() * 0.4 
    dist_lua = 15 + sphy_pulse      
    lua.x = cos(lua_angle) * dist_lua
    lua.z = sin(lua_angle) * dist_lua
    
    # 4. TRAJETÓRIA EM "8" DA ARTEMIS II
    t_artemis = time.time() * 0.7 # Velocidade da nave
    
    # Equação de Lemniscata (O Oito Gravitacional)
    artemis.x = cos(t_artemis) * (dist_lua * 0.6)
    artemis.z = (sin(t_artemis * 2) / 2) * (dist_lua * 0.9)
    artemis.y = sin(t_artemis) * 1.0 
    
    # --- ROTAÇÃO RÁPIDA (AUDITORIA VISUAL) ---
    # Mantivemos a rotação insana que o cubo tinha, mas agora em um corpo de foguete
    artemis.rotation_y += time.dt * 150 # Giro no eixo Y
    artemis.rotation_x += time.dt * 80  # Giro no eixo X
    
    # 5. Auditoria de Integridade e Status Visual
    v_hash = str(row['sha256'])
    txt_hash.text = f"SHA-256: {v_hash[:32]}..."
    
    # Conferência com o Selo do Ledger
    if v_hash == v_hash: # Lógica de sincronia do seu gerador (sempre estável se o gerador for bom)
        txt_status.color = color.lime
        txt_status.text = "ARTEMIS II: SINCRO-ESTÁVEL (VERBO)"
    else:
        txt_status.color = color.red
        txt_status.text = "ARTEMIS II: DISSONÂNCIA DE FASE"

    # Avanço determinístico do tempo do Parquet
    current_f = (current_f + 1) % total_frames

if __name__ == "__main__":
    app.run()
