import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsPolygonItem, QGraphicsTextItem,
                             QGraphicsEllipseItem, QGraphicsItem, QFrame, QSlider, QCheckBox, QGroupBox, QGridLayout, QRadioButton, QButtonGroup)
from PyQt6.QtGui import QPolygonF, QBrush, QColor, QPen, QFont, QPainter, QPixmap
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. VISUAL THEME
# ==========================================
GRID_SIZE = 6
CELL_SIZE = 110
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 60
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 400

C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_CELL_ZERO = QColor("#434C5E") 
C_LINE      = QColor("#4C566A") 
C_TEXT      = QColor("#ECEFF4") 
C_AGENT     = QColor("#EBCB8B") 
C_WALL      = QColor("#2E3440") 
C_POS_LOW, C_POS_HIGH = QColor("#5E81AC"), QColor("#A3BE8C") 
C_NEG_LOW, C_NEG_HIGH = QColor("#B48EAD"), QColor("#BF616A") 

def interpolate_color(c1, c2, ratio):
    r = int(c1.red() * (1 - ratio) + c2.red() * ratio)
    g = int(c1.green() * (1 - ratio) + c2.green() * ratio)
    b = int(c1.blue() * (1 - ratio) + c2.blue() * ratio)
    return QColor(r, g, b, 230)

def get_q_color(val):
    if val > 0.01: return interpolate_color(C_POS_LOW, C_POS_HIGH, min(1.0, val / 1.0))
    elif val < -0.01: return interpolate_color(C_NEG_LOW, C_NEG_HIGH, min(1.0, abs(val) / 1.0))
    return C_CELL_ZERO

# ==========================================
# 2. DUAL-MODE PHYSICS ENGINE
# ==========================================
class MarsRoverEngine:
    def __init__(self):
        self.size = GRID_SIZE
        self.start_pos = (0, 0)
        self.agent_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        self.pits = {(1, 3), (2, 3), (3, 3), (4, 1)}
        self.walls = {(2, 2)}
        self.Q = np.zeros((self.size, self.size, 4))
        
        # PARAMETERS
        self.gamma = 0.9
        self.epsilon = 0.1
        self.penalty = -0.04
        self.slippery = False  # The Physics Stochasticity
        self.mode_mdp = False  # Toggle: False=Q-Learning, True=Bellman Planning
        self.alpha = 0.5       # Learning Rate (Only for Q-Learning)

    def reset_brain(self):
        self.Q = np.zeros((self.size, self.size, 4))
        self.agent_pos = self.start_pos

    # --- HELPER: SIMULATE PHYSICS ---
    def get_next_state_reward(self, r, c, action):
        """Returns (nr, nc, reward, done) for a given deterministic move"""
        dr, dc = 0, 0
        if action == 0: dr = -1   # UP
        elif action == 1: dr = 1  # DOWN
        elif action == 2: dc = -1 # LEFT
        elif action == 3: dc = 1  # RIGHT
        
        nr, nc = r + dr, c + dc
        
        # Wall/Boundary Check
        if not (0 <= nr < self.size and 0 <= nc < self.size) or (nr, nc) in self.walls:
            nr, nc = r, c 
            
        reward = self.penalty
        done = False
        
        if (nr, nc) == self.goal_pos:
            reward = 1.0
            done = True
        elif (nr, nc) in self.pits:
            reward = -1.0
            done = True
            
        return nr, nc, reward, done

    # --- MODE 1: Q-LEARNING (Sample & Learn) ---
    def step_q_learning(self):
        r, c = self.agent_pos
        
        # Epsilon-Greedy
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0,1,2,3])
        else:
            action = np.argmax(self.Q[r, c, :] + np.random.randn(4)*0.001)

        # Apply Slippery Physics (Sampling)
        move_action = action
        if self.slippery and np.random.rand() > 0.8:
            if action in [0, 1]: move_action = np.random.choice([2, 3])
            else: move_action = np.random.choice([0, 1])

        # Execute
        nr, nc, reward, done = self.get_next_state_reward(r, c, move_action)
        
        # Q-Update (Temporal Difference)
        old_val = self.Q[r, c, action]
        next_max = 0 if done else np.max(self.Q[nr, nc, :])
        target = reward + self.gamma * next_max
        self.Q[r, c, action] = old_val + self.alpha * (target - old_val)
        
        self.agent_pos = (nr, nc)
        if done: self.agent_pos = self.start_pos # Auto reset position

    # --- MODE 2: MDP BELLMAN (Planning / Sweep) ---
    def step_bellman_sweep(self):
        """
        Runs ONE full pass of Value Iteration over the whole grid.
        Equation: Q(s,a) = Sum_s' P(s'|s,a) * [R + gamma * max_a' Q(s',a')]
        """
        new_Q = np.copy(self.Q)
        
        for r in range(self.size):
            for c in range(self.size):
                # Skip terminals (optional, but cleaner)
                if (r,c) == self.goal_pos or (r,c) in self.pits or (r,c) in self.walls:
                    continue
                    
                for action in [0,1,2,3]:
                    # CALCULATE EXPECTED VALUE
                    outcomes = []
                    
                    if self.slippery:
                        # 80% Intended, 10% Left slip, 10% Right slip
                        outcomes.append((0.8, action))
                        if action in [0,1]: # Up/Down
                            outcomes.append((0.1, 2)); outcomes.append((0.1, 3))
                        else: # Left/Right
                            outcomes.append((0.1, 0)); outcomes.append((0.1, 1))
                    else:
                        # 100% Intended
                        outcomes.append((1.0, action))
                        
                    # Summation
                    expected_value = 0
                    for prob, actual_move in outcomes:
                        nr, nc, reward, done = self.get_next_state_reward(r, c, actual_move)
                        
                        # Value of next state V(s') = max_a Q(s', a)
                        # If done, next value is 0
                        v_next = 0 if done else np.max(self.Q[nr, nc, :])
                        
                        expected_value += prob * (reward + self.gamma * v_next)
                    
                    new_Q[r, c, action] = expected_value

        self.Q = new_Q
        # Move Agent greedily just to show the policy
        self.move_agent_greedy()

    def move_agent_greedy(self):
        r, c = self.agent_pos
        action = np.argmax(self.Q[r, c, :])
        nr, nc, _, done = self.get_next_state_reward(r, c, action)
        self.agent_pos = (nr, nc)
        if done: self.agent_pos = self.start_pos

# ==========================================
# 3. CUSTOM ROBOT ITEM
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
            painter.setBrush(QBrush(QColor("#EBCB8B")))
            painter.setPen(QPen(QColor("white"), 3))
            painter.drawEllipse(10, 10, 60, 60)

# ==========================================
# 4. UI VISUALIZATION
# ==========================================
class GridCell:
    def __init__(self, r, c, scene):
        self.triangles = {}
        self.texts = {}
        x, y = c * CELL_SIZE, r * CELL_SIZE
        center = QPointF(x + CELL_SIZE/2, y + CELL_SIZE/2)
        
        coords = {
            0: [QPointF(x, y), QPointF(x + CELL_SIZE, y), center],
            1: [QPointF(x, y + CELL_SIZE), QPointF(x + CELL_SIZE, y + CELL_SIZE), center],
            2: [QPointF(x, y), QPointF(x, y + CELL_SIZE), center],
            3: [QPointF(x + CELL_SIZE, y), QPointF(x + CELL_SIZE, y + CELL_SIZE), center]
        }
        t_pos = {
            0: (x + CELL_SIZE/2 - 12, y + 5),
            1: (x + CELL_SIZE/2 - 12, y + CELL_SIZE - 20),
            2: (x + 5, y + CELL_SIZE/2 - 10),
            3: (x + CELL_SIZE - 30, y + CELL_SIZE/2 - 10)
        }

        for action in [0,1,2,3]:
            # Triangles
            poly = QPolygonF(coords[action])
            tri = QGraphicsPolygonItem(poly)
            tri.setPen(QPen(C_LINE, 1))
            tri.setBrush(QBrush(C_CELL_ZERO))
            scene.addItem(tri)
            self.triangles[action] = tri
            
            # Text
            txt = QGraphicsTextItem("")
            txt.setFont(QFont("Menlo", 9, QFont.Weight.Bold))
            txt.setDefaultTextColor(C_TEXT)
            txt.setPos(*t_pos[action])
            txt.setZValue(5)
            scene.addItem(txt)
            self.texts[action] = txt

    def update(self, q_values):
        for action in [0,1,2,3]:
            val = q_values[action]
            self.triangles[action].setBrush(QBrush(get_q_color(val)))
            if abs(val) < 0.01:
                self.texts[action].setPlainText("")
            else:
                self.texts[action].setPlainText(f"{val:.2f}")
                self.texts[action].setDefaultTextColor(QColor("white"))

    def set_static(self, color, label, scene):
        x = self.triangles[0].polygon().at(0).x()
        y = self.triangles[0].polygon().at(0).y()
        rect = scene.addRect(x, y, CELL_SIZE, CELL_SIZE)
        if label == "WALL": rect.setBrush(QBrush(color, Qt.BrushStyle.DiagCrossPattern))
        else: rect.setBrush(QBrush(color))
        rect.setPen(QPen(C_LINE, 2))
        if label:
            t = scene.addText(label)
            t.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            t.setDefaultTextColor(C_BG_DARK)
            br = t.boundingRect()
            t.setPos(x + (CELL_SIZE - br.width())/2, y + (CELL_SIZE - br.height())/2)

# ==========================================
# 5. MAIN APP
# ==========================================
class MarsRoverFinal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mars Rover: RL vs MDP")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"QMainWindow {{ background-color: {C_BG_DARK.name()}; }} QLabel {{ color: {C_TEXT.name()}; }}")
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Graphics
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(C_BG_DARK))
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 1px solid {C_LINE.name()}; border-radius: 4px;")
        layout.addWidget(self.view)
        
        self.engine = MarsRoverEngine()
        self.cells = {}
        self.init_graphics()
        
        # Controls
        self.create_control_deck(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)

    def create_control_deck(self, parent_layout):
        deck = QFrame()
        deck.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 10px; margin-top: 10px;")
        deck_layout = QVBoxLayout(deck)
        
        # --- MODE SELECTION ---
        mode_layout = QHBoxLayout()
        lbl_mode = QLabel("Algorithm Mode:")
        lbl_mode.setStyleSheet("font-weight: bold; font-size: 14px;")
        mode_layout.addWidget(lbl_mode)
        
        self.rb_q = QRadioButton("Q-Learning (Experience)")
        self.rb_q.setChecked(True)
        self.rb_q.setStyleSheet(f"color: {C_POS_HIGH.name()}; font-weight: bold;")
        self.rb_q.toggled.connect(self.switch_mode)
        
        self.rb_mdp = QRadioButton("Bellman MDP (Planning)")
        self.rb_mdp.setStyleSheet(f"color: {C_NEG_HIGH.name()}; font-weight: bold;")
        self.rb_mdp.toggled.connect(self.switch_mode)
        
        bg = QButtonGroup(self)
        bg.addButton(self.rb_q); bg.addButton(self.rb_mdp)
        mode_layout.addWidget(self.rb_q); mode_layout.addWidget(self.rb_mdp); mode_layout.addStretch()
        deck_layout.addLayout(mode_layout)

        # --- SLIDERS ---
        param_group = QGroupBox("Hyperparameters")
        param_group.setStyleSheet(f"QGroupBox {{ color: {C_TEXT.name()}; border: 1px solid {C_LINE.name()}; margin-top: 10px; }} QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }}")
        grid = QGridLayout(param_group)
        
        def add_slider(label, min_v, max_v, init_v, r, c, handler):
            txt_val = handler(init_v) # Get initial string
            l = QLabel(f"{label}: {txt_val}")
            s = QSlider(Qt.Orientation.Horizontal)
            s.setMinimum(min_v); s.setMaximum(max_v); s.setValue(int(init_v))
            s.valueChanged.connect(lambda v: l.setText(f"{label}: {handler(v)}"))
            grid.addWidget(l, r, c); grid.addWidget(s, r+1, c)
            return s

        # Handlers
        def h_gamma(v): self.engine.gamma = v/100.0; return f"{v/100.0:.2f}"
        def h_eps(v): self.engine.epsilon = v/100.0; return f"{v/100.0:.2f}"
        def h_pen(v): self.engine.penalty = -v/100.0; return f"{-v/100.0:.2f}"

        add_slider("Gamma (γ)", 10, 99, 90, 0, 0, h_gamma)
        add_slider("Exploration (ε)", 0, 100, 10, 0, 1, h_eps)
        add_slider("Living Cost", 0, 200, 4, 0, 2, h_pen)
        
        chk = QCheckBox("Slippery Physics")
        chk.setStyleSheet(f"color: {C_TEXT.name()}; font-weight: bold;")
        chk.toggled.connect(lambda c: setattr(self.engine, 'slippery', c))
        grid.addWidget(chk, 1, 3)
        deck_layout.addWidget(param_group)

        # --- PLAYBACK ---
        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Run")
        self.btn_play.setCheckable(True)
        self.btn_play.setStyleSheet(f"background-color: {C_POS_LOW.name()}; color: white; border-radius: 5px; padding: 10px; font-weight: bold;")
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.btn_reset = QPushButton("Reset Brain")
        self.btn_reset.setStyleSheet(f"background-color: {C_NEG_HIGH.name()}; color: white; border-radius: 5px; padding: 10px; font-weight: bold;")
        self.btn_reset.clicked.connect(self.reset)
        
        btn_layout.addWidget(self.btn_play); btn_layout.addWidget(self.btn_reset)
        deck_layout.addLayout(btn_layout)
        parent_layout.addWidget(deck)

    def init_graphics(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = GridCell(r, c, self.scene)
                if (r,c) == self.engine.goal_pos: cell.set_static(C_POS_HIGH, "GOAL", self.scene)
                elif (r,c) in self.engine.pits: cell.set_static(C_NEG_HIGH, "-1.0", self.scene)
                elif (r,c) in self.engine.walls: cell.set_static(C_WALL, "WALL", self.scene)
                else: self.cells[(r,c)] = cell
        
        self.robot = RobotItem()
        self.scene.addItem(self.robot)
        self.update_agent()

    def update_agent(self):
        r, c = self.engine.agent_pos
        self.robot.setPos(c * CELL_SIZE + 15, r * CELL_SIZE + 15)

    def switch_mode(self):
        self.engine.mode_mdp = self.rb_mdp.isChecked()
        self.reset()
        if self.engine.mode_mdp:
            self.btn_play.setText("▶ Run Sweep (Planning)")
        else:
            self.btn_play.setText("▶ Run Steps (Learning)")

    def tick(self):
        if self.engine.mode_mdp:
            self.engine.step_bellman_sweep()
        else:
            self.engine.step_q_learning()
            
        self.update_agent()
        for (r,c), cell in self.cells.items():
            cell.update(self.engine.Q[r,c])

    def toggle_play(self):
        if self.btn_play.isChecked():
            self.btn_play.setText("⏸ Pause")
            # MDP can run slower (150ms) to show the wave. Q-Learning faster (25ms).
            self.timer.start(150 if self.engine.mode_mdp else 25)
        else:
            self.btn_play.setText("▶ Resume")
            self.timer.stop()

    def reset(self):
        self.engine.reset_brain()
        for cell in self.cells.values(): cell.update(np.zeros(4))
        self.update_agent()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MarsRoverFinal()
    win.show()
    sys.exit(app.exec())