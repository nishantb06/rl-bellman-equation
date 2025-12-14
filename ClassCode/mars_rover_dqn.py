import sys
import os
import numpy as np
import random
from collections import deque

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.optim as optim

# --- PYQT ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsPolygonItem, QGraphicsTextItem,
                             QGraphicsItem, QFrame, QSlider, QCheckBox, QGroupBox, 
                             QGridLayout, QRadioButton, QButtonGroup)
from PyQt6.QtGui import QPolygonF, QBrush, QColor, QPen, QFont, QPainter, QPixmap
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. CONFIGURATION
# ==========================================
GRID_SIZE = 6
CELL_SIZE = 110
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 60
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 550 

# Nordic Theme Colors
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A") 
C_CELL_ZERO = QColor("#434C5E") 
C_LINE      = QColor("#4C566A") 
C_TEXT      = QColor("#ECEFF4") 
C_ACCENT    = QColor("#88C0D0") 

# --- FIXED MISSING CONSTANT ---
C_WALL      = QColor("#3B4252") 
# ------------------------------

# Gradients
C_POS_LOW, C_POS_HIGH = QColor("#5E81AC"), QColor("#A3BE8C") 
C_NEG_LOW, C_NEG_HIGH = QColor("#B48EAD"), QColor("#BF616A") 

def get_q_color(val):
    if val > 0.01: 
        ratio = min(1.0, val / 1.0)
        return QColor(
            int(C_POS_LOW.red()*(1-ratio) + C_POS_HIGH.red()*ratio),
            int(C_POS_LOW.green()*(1-ratio) + C_POS_HIGH.green()*ratio),
            int(C_POS_LOW.blue()*(1-ratio) + C_POS_HIGH.blue()*ratio), 230)
    elif val < -0.01: 
        ratio = min(1.0, abs(val) / 1.0)
        return QColor(
            int(C_NEG_LOW.red()*(1-ratio) + C_NEG_HIGH.red()*ratio),
            int(C_NEG_LOW.green()*(1-ratio) + C_NEG_HIGH.green()*ratio),
            int(C_NEG_LOW.blue()*(1-ratio) + C_NEG_HIGH.blue()*ratio), 230)
    return C_CELL_ZERO

# ==========================================
# 2. NEURAL NETWORK
# ==========================================
class SimpleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 3. PHYSICS ENGINE
# ==========================================
class MarsRoverEngine:
    def __init__(self):
        self.size = GRID_SIZE
        self.start_pos = (0, 0)
        self.agent_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        self.pits = {(1, 3), (2, 3), (3, 3), (4, 1)}
        self.walls = {(2, 2)}
        self.Q_table_vis = np.zeros((self.size, self.size, 4))
        
        # Hyperparameters
        self.gamma = 0.9
        self.epsilon = 0.1
        self.penalty = -0.04
        self.slippery = False
        self.mode = "Q" # Q, MDP, DQN
        self.alpha = 0.5 

        # DQN Stuff
        self.input_dim = self.size * self.size
        self.model = SimpleDQN(self.input_dim, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=500)
        self.batch_size = 32
        self.steps_done = 0

    def reset_brain(self):
        self.Q_table_vis = np.zeros((self.size, self.size, 4))
        self.agent_pos = self.start_pos
        self.model = SimpleDQN(self.input_dim, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.replay_buffer.clear()

    def get_state_vector(self, r, c):
        vec = np.zeros(self.input_dim)
        vec[r * self.size + c] = 1.0
        return torch.FloatTensor(vec)

    def get_next_state_reward(self, r, c, action):
        dr, dc = 0, 0
        if action == 0: dr = -1
        elif action == 1: dr = 1
        elif action == 2: dc = -1
        elif action == 3: dc = 1
        nr, nc = r + dr, c + dc
        
        if not (0 <= nr < self.size and 0 <= nc < self.size) or (nr, nc) in self.walls:
            nr, nc = r, c 
            
        reward = self.penalty 
        done = False
        if (nr, nc) == self.goal_pos: reward = 1.0; done = True
        elif (nr, nc) in self.pits: reward = -1.0; done = True
        return nr, nc, reward, done

    # --- ALGORITHMS ---
    def step_q_learning(self):
        r, c = self.agent_pos
        if np.random.rand() < self.epsilon: action = np.random.choice(4)
        else: action = np.argmax(self.Q_table_vis[r, c, :] + np.random.randn(4)*0.001)

        move_action = self._apply_slip(action)
        nr, nc, reward, done = self.get_next_state_reward(r, c, move_action)
        
        old = self.Q_table_vis[r, c, action]
        nxt = 0 if done else np.max(self.Q_table_vis[nr, nc, :])
        self.Q_table_vis[r, c, action] = old + self.alpha * (reward + self.gamma * nxt - old)
        
        self.agent_pos = (nr, nc)
        if done: self.agent_pos = self.start_pos

    def step_bellman(self):
        new_Q = np.copy(self.Q_table_vis)
        for r in range(self.size):
            for c in range(self.size):
                if (r,c) == self.goal_pos or (r,c) in self.pits or (r,c) in self.walls: continue
                for a in range(4):
                    out = self._get_outcomes(a)
                    ev = 0
                    for p, m in out:
                        nr, nc, rew, done = self.get_next_state_reward(r, c, m)
                        v_next = 0 if done else np.max(self.Q_table_vis[nr, nc, :])
                        ev += p * (rew + self.gamma * v_next)
                    new_Q[r, c, a] = ev
        self.Q_table_vis = new_Q
        r,c = self.agent_pos
        a = np.argmax(self.Q_table_vis[r, c, :])
        nr, nc, _, done = self.get_next_state_reward(r, c, a)
        self.agent_pos = (nr, nc)
        if done: self.agent_pos = self.start_pos

    def step_dqn(self):
        r, c = self.agent_pos
        state = self.get_state_vector(r, c)
        
        if np.random.rand() < self.epsilon: action = np.random.choice(4)
        else:
            with torch.no_grad(): action = torch.argmax(self.model(state)).item()

        move_action = self._apply_slip(action)
        nr, nc, reward, done = self.get_next_state_reward(r, c, move_action)
        
        self.replay_buffer.append((r, c, action, reward, nr, nc, done))
        
        if len(self.replay_buffer) > self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            states = torch.stack([self.get_state_vector(x[0], x[1]) for x in batch])
            next_s = torch.stack([self.get_state_vector(x[4], x[5]) for x in batch])
            acts = torch.tensor([x[2] for x in batch])
            rews = torch.tensor([x[3] for x in batch], dtype=torch.float32)
            dones = torch.tensor([x[6] for x in batch], dtype=torch.float32)

            curr_Q = self.model(states).gather(1, acts.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next = self.model(next_s).max(1)[0]
                target = rews + self.gamma * max_next * (1 - dones)

            loss = self.criterion(curr_Q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % 5 == 0: self._update_vis_from_nn()
        self.agent_pos = (nr, nc)
        if done: self.agent_pos = self.start_pos

    def _update_vis_from_nn(self):
        with torch.no_grad():
            for r in range(self.size):
                for c in range(self.size):
                    self.Q_table_vis[r, c, :] = self.model(self.get_state_vector(r, c)).numpy()

    def _apply_slip(self, a):
        if not self.slippery or np.random.rand() < 0.8: return a
        return np.random.choice([2,3] if a in [0,1] else [0,1])

    def _get_outcomes(self, a):
        if not self.slippery: return [(1.0, a)]
        return [(0.8, a)] + ([(0.1, 2), (0.1, 3)] if a in [0,1] else [(0.1, 0), (0.1, 1)])

# ==========================================
# 4. CUSTOM ROBOT ITEM
# ==========================================
class RobotItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(200)
        self.pixmap = None
        if os.path.exists("robot.png"):
            self.pixmap = QPixmap("robot.png")
            self.pixmap = self.pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def boundingRect(self):
        return QRectF(0, 0, 80, 80)

    def paint(self, painter, option, widget):
        if self.pixmap:
            off_x = (80 - self.pixmap.width()) / 2
            off_y = (80 - self.pixmap.height()) / 2
            painter.drawPixmap(int(off_x), int(off_y), self.pixmap)
        else:
            painter.setBrush(QBrush(QColor("#5E81AC")))
            painter.setPen(QPen(QColor("white"), 3))
            painter.drawEllipse(10, 10, 60, 60)

# ==========================================
# 5. MAIN UI
# ==========================================
class MarsRoverUltimate(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Studio: Tabular vs DQN")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"QMainWindow {{ background-color: {C_BG_DARK.name()}; }} QLabel {{ color: {C_TEXT.name()}; font-family: Arial; }}")
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(C_BG_DARK))
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 1px solid {C_LINE.name()}; border-radius: 4px;")
        layout.addWidget(self.view)
        
        self.engine = MarsRoverEngine()
        self.cells = {}
        self.init_graphics()
        self.create_controls(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.update_insights()

    def create_controls(self, layout):
        deck = QFrame()
        deck.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 10px; margin-top: 5px;")
        vbox = QVBoxLayout(deck)
        
        hbox_mode = QHBoxLayout()
        self.rb_q = QRadioButton("Tabular Q-Learning"); self.rb_q.setChecked(True)
        self.rb_mdp = QRadioButton("Bellman MDP")
        self.rb_dqn = QRadioButton("Deep Q-Network")
        
        self.rb_q.setStyleSheet(f"color: {C_POS_HIGH.name()}; font-weight: bold;")
        self.rb_mdp.setStyleSheet(f"color: #D08770; font-weight: bold;")
        self.rb_dqn.setStyleSheet(f"color: #EBCB8B; font-weight: bold;")
        
        bg = QButtonGroup(self)
        bg.addButton(self.rb_q); bg.addButton(self.rb_mdp); bg.addButton(self.rb_dqn)
        hbox_mode.addWidget(self.rb_q); hbox_mode.addWidget(self.rb_mdp); hbox_mode.addWidget(self.rb_dqn)
        vbox.addLayout(hbox_mode)
        
        self.rb_q.toggled.connect(lambda: self.switch_mode("Q"))
        self.rb_mdp.toggled.connect(lambda: self.switch_mode("MDP"))
        self.rb_dqn.toggled.connect(lambda: self.switch_mode("DQN"))

        self.lbl_insight = QLabel("Info")
        self.lbl_insight.setWordWrap(True)
        self.lbl_insight.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px; font-style: italic; font-size: 13px;")
        vbox.addWidget(self.lbl_insight)

        grid = QGridLayout()
        def add_sl(lbl, min, max, init, r, c, fn):
            l = QLabel(f"{lbl}: {fn(init)}")
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(min, max); s.setValue(init)
            s.valueChanged.connect(lambda v: (fn(v), l.setText(f"{lbl}: {fn(v)}")))
            grid.addWidget(l, r, c); grid.addWidget(s, r+1, c)

        add_sl("Gamma", 10, 99, 90, 0, 0, lambda v: setattr(self.engine, 'gamma', v/100) or f"{v/100:.2f}")
        add_sl("Epsilon", 0, 100, 10, 0, 1, lambda v: setattr(self.engine, 'epsilon', v/100) or f"{v/100:.2f}")
        add_sl("Living Cost", 0, 200, 4, 0, 2, lambda v: setattr(self.engine, 'penalty', -v/100) or f"{-v/100:.2f}")

        self.btn_play = QPushButton("▶ Run"); self.btn_play.setCheckable(True)
        self.btn_play.setStyleSheet(f"background-color: {C_POS_LOW.name()}; color: white; padding: 8px;")
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.btn_rst = QPushButton("Reset Brain"); self.btn_rst.setStyleSheet(f"background-color: {C_NEG_HIGH.name()}; color: white; padding: 8px;")
        self.btn_rst.clicked.connect(self.reset)
        
        grid.addWidget(self.btn_play, 0, 3); grid.addWidget(self.btn_rst, 1, 3)
        vbox.addLayout(grid)
        layout.addWidget(deck)

    def switch_mode(self, mode):
        self.engine.mode = mode
        self.reset()
        self.update_insights()

    def update_insights(self):
        m = self.engine.mode
        if m == "Q":
            txt = ("<b>Tabular Q-Learning:</b> Updates <b>one cell at a time</b> based on experience. "
                   "Map looks 'patchy' because values only change where the agent walks.")
        elif m == "MDP":
            txt = ("<b>Bellman Planning:</b> Global calculation sweep. Values spread like a <b>perfect wave</b> from the goal instantly.")
        elif m == "DQN":
            txt = ("<b>Deep Q-Network:</b> Finds goal once -> <b>Entire map</b> might change color! "
                   "NN generalizes inputs, making it powerful but unstable (watch for sudden 'Red Flashes').")
        self.lbl_insight.setText(txt)

    def init_graphics(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self.cells[(r,c)] = GridCell(r, c, self.scene)
                if (r,c)==self.engine.goal_pos: self.cells[(r,c)].set_static(C_POS_HIGH, "GOAL", self.scene)
                elif (r,c) in self.engine.pits: self.cells[(r,c)].set_static(C_NEG_HIGH, "-1", self.scene)
                elif (r,c) in self.engine.walls: self.cells[(r,c)].set_static(C_WALL, "WALL", self.scene)
        
        self.robot = RobotItem()
        self.scene.addItem(self.robot)
        self.update_agent()

    def update_agent(self):
        r, c = self.engine.agent_pos
        self.robot.setPos(c * CELL_SIZE + 15, r * CELL_SIZE + 15)

    def tick(self):
        m = self.engine.mode
        if m == "Q": self.engine.step_q_learning()
        elif m == "MDP": self.engine.step_bellman()
        elif m == "DQN": self.engine.step_dqn()
        
        self.update_agent()
        for (r,c), cell in self.cells.items(): cell.update(self.engine.Q_table_vis[r,c])

    def toggle_play(self):
        if self.btn_play.isChecked():
            self.btn_play.setText("⏸ Pause")
            self.timer.start(150 if self.engine.mode == "MDP" else 30)
        else:
            self.btn_play.setText("▶ Resume"); self.timer.stop()

    def reset(self):
        self.engine.reset_brain()
        for cell in self.cells.values(): cell.update(np.zeros(4))
        self.update_agent()

class GridCell:
    def __init__(self, r, c, scene):
        self.triangles, self.texts = {}, {}
        x, y = c * CELL_SIZE, r * CELL_SIZE
        center = QPointF(x + CELL_SIZE/2, y + CELL_SIZE/2)
        coords = {
            0: [QPointF(x, y), QPointF(x + CELL_SIZE, y), center],
            1: [QPointF(x, y + CELL_SIZE), QPointF(x + CELL_SIZE, y + CELL_SIZE), center],
            2: [QPointF(x, y), QPointF(x, y + CELL_SIZE), center],
            3: [QPointF(x + CELL_SIZE, y), QPointF(x + CELL_SIZE, y + CELL_SIZE), center]
        }
        t_pos = {
            0: (x + CELL_SIZE/2 - 12, y + 5), 1: (x + CELL_SIZE/2 - 12, y + CELL_SIZE - 20),
            2: (x + 5, y + CELL_SIZE/2 - 10), 3: (x + CELL_SIZE - 30, y + CELL_SIZE/2 - 10)
        }
        for a in range(4):
            tri = QGraphicsPolygonItem(QPolygonF(coords[a]))
            tri.setPen(QPen(C_LINE, 1)); tri.setBrush(QBrush(C_CELL_ZERO))
            scene.addItem(tri); self.triangles[a] = tri
            txt = QGraphicsTextItem(""); txt.setFont(QFont("Menlo", 9, QFont.Weight.Bold)); txt.setDefaultTextColor(C_TEXT)
            txt.setPos(*t_pos[a]); txt.setZValue(5); scene.addItem(txt); self.texts[a] = txt

    def update(self, q_values):
        for a in range(4):
            val = q_values[a]
            self.triangles[a].setBrush(QBrush(get_q_color(val)))
            self.texts[a].setPlainText(f"{val:.2f}" if abs(val) > 0.01 else "")
            if abs(val) > 0.1: self.texts[a].setDefaultTextColor(QColor("white"))

    def set_static(self, color, label, scene):
        x = self.triangles[0].polygon().at(0).x()
        y = self.triangles[0].polygon().at(0).y()
        rect = scene.addRect(x, y, CELL_SIZE, CELL_SIZE)
        rect.setBrush(QBrush(color, Qt.BrushStyle.DiagCrossPattern if label=="WALL" else Qt.BrushStyle.SolidPattern))
        rect.setPen(QPen(C_LINE, 2))
        if label:
            t = scene.addText(label)
            t.setFont(QFont("Arial", 12, QFont.Weight.Bold)); t.setDefaultTextColor(C_BG_DARK)
            br = t.boundingRect()
            t.setPos(x + (CELL_SIZE - br.width())/2, y + (CELL_SIZE - br.height())/2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MarsRoverUltimate()
    win.show()
    sys.exit(app.exec())