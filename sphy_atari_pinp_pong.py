import py5
import pandas as pd
import numpy as np

# --- CONFIGURAÇÕES TÉCNICAS ---
WIDTH, HEIGHT = 1000, 600
P_WIDTH, P_HEIGHT = 15, 100
MAX_FRAMES = 1200 # Limite da existência do Parquet

class SphyPong:
    def __init__(self):
        print("🌌 Sincronizando com o tecido do Universo... (Carregando Parquet)")
        self.df = pd.read_parquet("sphy_frames.parquet")
        self.total_frames = len(self.df)
        self.frame_idx = 0
        self.game_over = False
        
        # Elementos do Jogo
        self.ball_x, self.ball_y = WIDTH//2, HEIGHT//2
        self.ball_vx, self.ball_vy = 8, 6
        self.universe_y = HEIGHT//2
        
        # Placar
        self.score_deywe = 0
        self.score_universe = 0

    def update(self):
        # TRAVA DIMENSIONAL: Se chegar em 1200, o jogo para
        if self.frame_idx >= MAX_FRAMES:
            self.game_over = True
            return

        # 1. Movimentação da Bola
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # 2. IA UNIVERSE (Baseada no Parquet)
        row_data = self.df.iloc[self.frame_idx]
        wave = np.array(row_data['wave_flat'])
        wave_map = wave.reshape(200, 100)
        
        # O Universe prevê a bola na sua coluna (99)
        col_idx = 99
        pressao_local = wave_map[:, col_idx]
        ponto_focal = np.argmax(pressao_local)
        alvo_y = py5.remap(ponto_focal, 0, 200, 0, HEIGHT)
        
        self.universe_y += (alvo_y - self.universe_y) * 0.15
        
        # 3. Colisões
        if self.ball_y < 0 or self.ball_y > HEIGHT:
            self.ball_vy *= -1
            
        # 4. Colisão: DEYWE (Esquerda)
        if self.ball_x < 40:
            if abs(self.ball_y - py5.mouse_y) < P_HEIGHT/2:
                self.ball_vx = abs(self.ball_vx) * 1.05
                self.ball_x = 40
            elif self.ball_x < 0:
                self.score_universe += 1
                self.reset_ball()

        # 5. Colisão: UNIVERSE (Direita)
        if self.ball_x > WIDTH - 40:
            if abs(self.ball_y - self.universe_y) < P_HEIGHT/2:
                self.ball_vx = -abs(self.ball_vx) * 1.05
                self.ball_x = WIDTH - 40
            elif self.ball_x > WIDTH:
                self.score_deywe += 1
                self.reset_ball()

        self.frame_idx += 1

    def reset_ball(self):
        self.ball_x, self.ball_y = WIDTH//2, HEIGHT//2
        self.ball_vx = 8 if self.ball_vx < 0 else -8
        self.ball_vy = 6

game = SphyPong()

def setup():
    py5.size(WIDTH, HEIGHT, py5.P2D)
    py5.text_align(py5.CENTER)

def draw():
    py5.background(5, 5, 15)
    
    # HUD Superior
    py5.fill(255)
    py5.text_size(24)
    py5.text(f"DEYWE: {game.score_deywe}", WIDTH * 0.25, 40)
    py5.text(f"UNIVERSE: {game.score_universe}", WIDTH * 0.75, 40)
    
    # Barra de Progresso do Tempo (Frame count)
    py5.no_stroke()
    py5.fill(50, 50, 100)
    py5.rect(0, HEIGHT-10, WIDTH, 10)
    py5.fill(0, 255, 255)
    progress = py5.remap(game.frame_idx, 0, MAX_FRAMES, 0, WIDTH)
    py5.rect(0, HEIGHT-10, progress, 10)

    if not game.game_over:
        game.update()
        
        # Raquete DEYWE (Azul)
        py5.fill(0, 200, 255)
        py5.rect(20, py5.mouse_y - P_HEIGHT/2, P_WIDTH, P_HEIGHT)
        
        # Raquete UNIVERSE (Rosa/Magenta)
        py5.fill(255, 0, 150)
        py5.rect(WIDTH - 35, game.universe_y - P_HEIGHT/2, P_WIDTH, P_HEIGHT)
        
        # Bola SPHY
        py5.fill(255, 255, 0)
        py5.ellipse(game.ball_x, game.ball_y, 20, 20)
        
        py5.fill(255, 150)
        py5.text(f"TEMPO SPHY: {game.frame_idx}", WIDTH//2, 40)
    
    else:
        # TELA DE ENCERRAMENTO (ESTADO ESTACIONÁRIO)
        py5.fill(0, 0, 0, 180)
        py5.rect(0, 0, WIDTH, HEIGHT)
        
        py5.fill(0, 255, 255)
        py5.text_size(50)
        py5.text("EVENTO DE HORIZONTE ALCANÇADO", WIDTH//2, HEIGHT//2 - 40)
        
        py5.text_size(30)
        py5.fill(255)
        py5.text(f"Você duelou contra o Universo por {MAX_FRAMES} frames", WIDTH//2, HEIGHT//2 + 20)
        
        ganhador = "DEYWE VENCEU O COSMOS!" if game.score_deywe > game.score_universe else "O UNIVERSE RECLAMOU A MATÉRIA!"
        if game.score_deywe == game.score_universe: ganhador = "EQUILÍBRIO UNIVERSAL ALCANÇADO"
        
        py5.fill(255, 255, 0)
        py5.text(ganhador, WIDTH//2, HEIGHT//2 + 80)
        py5.text_size(15)
        py5.text("O Parquet SPHY esgotou sua energia temporal.", WIDTH//2, HEIGHT//2 + 120)

if __name__ == "__main__":
    py5.run_sketch()
