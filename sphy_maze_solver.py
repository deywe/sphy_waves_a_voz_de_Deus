import py5
import pandas as pd
import numpy as np

# --- CONFIGURAÇÕES DO GRID ---
COLS, ROWS = 80, 40
CELL_SIZE = 20 

class SPHYMaze4D:
    def __init__(self):
        self.grid = self.gerar_maze(COLS, ROWS)
        print("📂 Carregando Hiperespaço Parquet...")
        self.df = pd.read_parquet("sphy_frames.parquet")
        self.total_frames = len(self.df)
        self.pos_ia = [0, 0] 
        self.caminho_3d = [(0, 0, 0)] 
        self.frame_idx = 0
        self.angle_y = 0

    def gerar_maze(self, c, r):
        np.random.seed(42)
        return np.random.choice([0, 1], size=(r, c), p=[0.85, 0.15])

    def resolver_passo(self):
        if self.frame_idx >= self.total_frames: return
        
        row_data = self.df.iloc[self.frame_idx]
        wave = np.array(row_data['wave_flat'])
        
        r, c = int(self.pos_ia[0]), int(self.pos_ia[1])
        vizinhos = [(r, c+1), (r+1, c), (r, c-1), (r-1, c)]
        
        melhor_v = None
        menor_p = float('inf')
        
        # 4ª Dimensão (Pulsação Temporal)
        t_factor = np.sin(self.frame_idx * 0.1)

        for vr, vc in vizinhos:
            if 0 <= vr < ROWS and 0 <= vc < COLS:
                if self.grid[vr, vc] == 0:
                    p_row = int(py5.remap(vr, 0, ROWS, 0, 199))
                    p_col = int(py5.remap(vc, 0, COLS, 0, 99))
                    idx = p_row * 100 + p_col
                    
                    z_fase = wave[idx] * 150 
                    
                    # Métrica de Minkowski simplificada (X, Y, Z, T)
                    dist_objetivo = np.sqrt((ROWS-1 - vr)**2 + (COLS-1 - vc)**2)
                    p_total = (z_fase * t_factor) + (dist_objetivo * 0.5)
                    
                    if p_total < menor_p:
                        menor_p = p_total
                        melhor_v = (vr, vc, z_fase)
        
        if melhor_v:
            self.pos_ia = [melhor_v[0], melhor_v[1]]
            self.caminho_3d.append(melhor_v)
        self.frame_idx += 1

maze = SPHYMaze4D()

def setup():
    py5.size(1200, 800, py5.P3D)
    py5.frame_rate(60)

def draw():
    py5.background(5, 5, 15)
    py5.ambient_light(100, 100, 100)
    py5.directional_light(255, 255, 255, 0, 1, -1)
    
    # Câmera Hiperdimensional
    py5.translate(py5.width/2, py5.height/2, -100)
    py5.rotate_x(py5.PI/3.5) 
    maze.angle_y += 0.005
    py5.rotate_z(maze.angle_y) 
    py5.translate(-COLS*CELL_SIZE/2, -ROWS*CELL_SIZE/2)

    # 1. TERRENO 4D (Correção do IndexError aqui)
    row_data = maze.df.iloc[maze.frame_idx % maze.total_frames]
    wave = np.array(row_data['wave_flat'])
    
    py5.no_stroke()
    for r in range(ROWS - 1): # Garante que r+1 não estoure
        py5.begin_shape(py5.TRIANGLE_STRIP)
        for c in range(COLS):
            for nr in [r, r+1]:
                p_row = int(py5.remap(nr, 0, ROWS, 0, 199))
                p_col = int(py5.remap(c, 0, COLS, 0, 99))
                z = wave[p_row * 100 + p_col] * 100
                
                if maze.grid[nr, c] == 1:
                    py5.fill(10, 10, 20) # Parede (Massa escura)
                    z = 40 # Paredes são elevadas
                else:
                    # Azul dinâmico conforme a profundidade Z
                    py5.fill(30, 60, py5.remap(z, -100, 100, 100, 255), 180)
                
                py5.vertex(c * CELL_SIZE, nr * CELL_SIZE, z)
        py5.end_shape()

    # 2. RESOLUÇÃO SPHY
    maze.resolver_passo()
    
    # Rastro Geodésico
    py5.no_fill()
    py5.stroke(0, 255, 255)
    py5.stroke_weight(3)
    py5.begin_shape()
    for p in maze.caminho_3d:
        py5.vertex(p[1] * CELL_SIZE, p[0] * CELL_SIZE, p[2] + 5)
    py5.end_shape()

    # 3. A IA (Partícula)
    py5.push_matrix()
    z_atual = maze.caminho_3d[-1][2]
    py5.translate(maze.pos_ia[1] * CELL_SIZE, maze.pos_ia[0] * CELL_SIZE, z_atual + 15)
    py5.fill(255, 255, 0)
    py5.no_stroke()
    py5.sphere(8)
    py5.pop_matrix()

if __name__ == "__main__":
    py5.run_sketch()
