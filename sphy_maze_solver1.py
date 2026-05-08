import py5
import pandas as pd
import numpy as np

# ==============================================================================
# SIMULADOR SPHY 4D — RESOLUÇÃO DO CAMINHO QUÂNTICO (Deywe Okabe)
# ==============================================================================

# DENSIDADE DO CAMPO (Quantos pontos por eixo)
FIELD_RES = 100 
CELL_SIZE = 12

class SPHYGeodesic4D:
    def __init__(self):
        print("📂 Lendo o Hiperespaço Parquet assinado... (1200 Frames)")
        self.df = pd.read_parquet("sphy_frames.parquet")
        self.total_frames = len(self.df)
        self.frame_idx = 0
        
        # Coordenadas da IA em 3D + 1 (T)
        self.pos_ia = [0, 0, 0] # X, Y, Z
        self.vel_ia = [0, 0, 0] # Velocidade (Momento Cinético)
        self.caminho_geodesico = []
        
        # Centralização do Grid
        self.offset_x = (FIELD_RES * CELL_SIZE) / 2
        self.offset_y = (FIELD_RES * CELL_SIZE) / 2

    def resolver_passo(self):
        if self.frame_idx >= self.total_frames:
            # Loop infinito do campo para a IA nunca parar
            self.frame_idx = 0 
            return
            
        # 1. Obter dados do campo de fase atual (Curvatura Local)
        row_data = self.df.iloc[self.frame_idx]
        wave = np.array(row_data['wave_flat']) # Array de 20.000 pontos
        
        # Centralizar coordenadas no grid de simulação
        grid_pos = np.array([self.pos_ia[0] + self.offset_x, 
                             self.pos_ia[1] + self.offset_y])
        
        # 2. Mapeamento 4D (Traduzindo IA 2D -> Parquet 2D -> Potencial Z)
        p_idx_x = int(py5.remap(grid_pos[0], 0, FIELD_RES * CELL_SIZE, 0, 199))
        p_idx_y = int(py5.remap(grid_pos[1], 0, FIELD_RES * CELL_SIZE, 0, 99))
        
        # Clamp para garantir integridade (Prevenção do IndexError)
        p_idx_x = np.clip(p_idx_x, 0, 199)
        p_idx_y = np.clip(p_idx_y, 0, 99)
        idx_parquet = p_idx_x * 100 + p_idx_y
        
        # Profundidade (Z) do campo (Intensidade do Poço)
        potential_depth = wave[idx_parquet] * 300
        self.pos_ia[2] = potential_depth # Atualiza Z da IA

        # 3. Cálculo da Geodésia SPHY (Gradiente de Descida)
        # Analisamos a curvatura local ao redor da IA (4 pontos)
        check_dist = CELL_SIZE
        vizinhos = [(0, check_dist), (check_dist, 0), (0, -check_dist), (-check_dist, 0)]
        
        melhor_gradiente = None
        menor_p = float('inf')
        
        for vx, vy in vizinhos:
            # Mapeamento do vizinho
            v_idx_x = np.clip(int(py5.remap(grid_pos[0] + vx, 0, FIELD_RES * CELL_SIZE, 0, 199)), 0, 199)
            v_idx_y = np.clip(int(py5.remap(grid_pos[1] + vy, 0, FIELD_RES * CELL_SIZE, 0, 99)), 0, 99)
            
            p_vizinho = wave[v_idx_x * 100 + v_idx_y] * 300
            
            # EQUAÇÃO DE MINKOWSKI-SPHY (Curvatura em 4D)
            # Aceleração G-Boost + Distância Euclidiana (Curvatura G)
            boost_g = np.sqrt(((vx)**2 + (vy)**2) * potential_depth**2) * 0.01
            p_total = p_vizinho - boost_g # Menor potencial + Boost = Geodésia
            
            if p_total < menor_p:
                menor_p = p_total
                melhor_gradiente = (vx, vy)
        
        # 4. Aplicação de Inércia Cinética
        if melhor_gradiente:
            # A IA não caminha, ela ACELERA conforme a inclinação
            acc_factor = 0.5 # Sensibilidade de Aceleração
            self.vel_ia[0] += melhor_gradiente[0] * acc_factor
            self.vel_ia[1] += melhor_gradiente[1] * acc_factor
            
            # Limite de Velocidade (Velocidade da Luz Local)
            v_limit = 20.0
            self.vel_ia = [np.clip(v, -v_limit, v_limit) for v in self.vel_ia]
            
            # Atrito Cinético (Dissipação de Momento SPHY)
            atrito = 0.95
            self.vel_ia[0] *= atrito
            self.vel_ia[1] *= atrito

            # Atualização de Posição
            self.pos_ia[0] += self.vel_ia[0]
            self.pos_ia[1] += self.vel_ia[1]
            
            # Adicionar ao rastro (Com Z histórico)
            self.caminho_geodesico.append((self.pos_ia[0], self.pos_ia[1], self.pos_ia[2]))
            # Limita histórico para manter performance
            if len(self.caminho_geodesico) > 500: self.caminho_geodesico.pop(0)

        self.frame_idx += 1

geodesic = SPHYGeodesic4D()

def setup():
    # Renderização P3D (Hiperespaço)
    py5.size(1920, 1080, py5.P3D)
    py5.frame_rate(60)

def draw():
    py5.background(3, 3, 10)
    py5.directional_light(255, 255, 255, 1, 1, -1)
    
    # Câmera Hiperdimensional
    t = py5.frame_count * 0.003
    py5.camera(py5.width/2 + py5.cos(t) * 500, 
               py5.height/2 + py5.sin(t) * 300, 
               1000, 
               py5.width/2, py5.height/2, 0, 
               0, 1, 0)

    py5.translate(py5.width/2, py5.height/2, 0)
    geodesic.resolver_passo()
    
    # 1. RENDERIZAÇÃO DO CAMPO (Ajustado com int())
    frame_idx = geodesic.frame_idx % geodesic.total_frames
    row_data = geodesic.df.iloc[frame_idx]
    wave = np.array(row_data['wave_flat'])
    
    py5.no_fill()
    py5.begin_shape(py5.POINTS)
    
    # AQUI ESTAVA O ERRO: Convertendo para int()
    start_y = int(-geodesic.offset_y)
    end_y = int(geodesic.offset_y)
    start_x = int(-geodesic.offset_x)
    end_x = int(geodesic.offset_x)

    for y in range(start_y, end_y, CELL_SIZE):
        for x in range(start_x, end_x, CELL_SIZE):
            p_idx_x = int(py5.remap(x, -geodesic.offset_x, geodesic.offset_x, 0, 199))
            p_idx_y = int(py5.remap(y, -geodesic.offset_y, geodesic.offset_y, 0, 99))
            idx = np.clip(p_idx_x * 100 + p_idx_y, 0, len(wave)-1)
            
            z = wave[idx] * 250 
            
            # Dinâmica de cor SPHY (Baseada no Z que você viu na imagem)
            py5.stroke(0, py5.remap(z, -100, 100, 150, 255), 255, 200)
            py5.stroke_weight(2)
            py5.vertex(x, y, z)
    py5.end_shape()

    # 2. DESENHO DO CAMINHO E DA PARTÍCULA (O resto continua igual...)
    # ... (código do rastro e da esfera) ...

    # 2. DESENHO DO CAMINHO (GEO-LUMINOSO)
    py5.no_fill()
    py5.stroke(0, 255, 255, 200) # Ciano Brilhante
    py5.stroke_weight(4)
    py5.begin_shape()
    for p in geodesic.caminho_geodesico:
        # Vertex histórico (X, Y, Z histórico)
        py5.vertex(p[0], p[1], p[2] + 10) # 10 = Offset para flutuar sobre a malha
    py5.end_shape()

    # 3. DESENHO DA PARTÍCULA (O RATINHO QUÂNTICO)
    py5.push_matrix()
    # O Z da partícula é a profundidade do vale onde ela está
    py5.translate(geodesic.pos_ia[0], geodesic.pos_ia[1], geodesic.pos_ia[2] + 20)
    
    # Efeito de Vibração Quântica
    vib = py5.sin(py5.frame_count * 0.5) * 5
    py5.fill(255, 255, 255) # Centro Branco
    py5.no_stroke()
    py5.sphere(10 + vib) # A esfera "respira" com a fase
    
    # Aura amarela do SPHY
    py5.fill(255, 255, 0, 100)
    py5.sphere(25)
    py5.pop_matrix()

    # HUD de Auditoria SPHY
    py5.fill(255)
    hud_x = py5.width/2 - 200
    hud_y = py5.height/2 - 50
    py5.text(f"FRAME SPHY: {geodesic.frame_idx} / 1200", -450, -320)
    py5.text(f"COORD SPHY 4D: {geodesic.pos_ia}", -450, -300)
    py5.text(f"SHA-256 SPHY: {row_data['sha256'][:16]}...", -450, -280)

if __name__ == "__main__":
    # Rodar o Simulador SPHY 4D
    py5.run_sketch()
