import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsPolygonItem, QGraphicsTextItem,
                             QGraphicsEllipseItem, QGraphicsItem, QFrame, QSlider, QCheckBox, QGroupBox, QGridLayout)
from PyQt6.QtGui import QPolygonF, QBrush, QColor, QPen, QFont, QPainter, QPixmap
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. VISUAL THEME: NORDIC POLAR
# ==========================================
GRID_SIZE = 6
CELL_SIZE = 110
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 60
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 350 # Extra space for Lab Controls

# Colors (Nord Theme)
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_CELL_ZERO = QColor("#434C5E") 
C_LINE      = QColor("#4C566A") 
C_TEXT      = QColor("#ECEFF4") 
C_AGENT     = QColor("#EBCB8B") 
C_WALL      = QColor("#2E3440") 

# Gradient Endpoints
C_POS_LOW, C_POS_HIGH = QColor("#5E81AC"), QColor("#A3BE8C") 
C_NEG_LOW, C_NEG_HIGH = QColor("#B48EAD"), QColor("#BF616A") 

def interpolate_color(c1, c2, ratio):
    """Blends two colors based on ratio 0.0 to 1.0"""
    r = int(c1.red() * (1 - ratio) + c2.red() * ratio)
    g = int(c1.green() * (1 - ratio) + c2.green() * ratio)
    b = int(c1.blue() * (1 - ratio) + c2.blue() * ratio)
    return QColor(r, g, b, 230)

def get_q_color(val):
    """Returns color based on Q-value"""
    if val > 0.01:
        # Scale 0 to 1.0
        return interpolate_color(C_POS_LOW, C_POS_HIGH, min(1.0, val / 1.0))
    elif val < -0.01:
        # Scale 0 to -1.0
        return interpolate_color(C_NEG_LOW, C_NEG_HIGH, min(1.0, abs(val) / 1.0))
    return C_CELL_ZERO

# ==========================================
# 2. PHYSICS ENGINE (With Lab Parameters)
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
        self.is_game_over = False
        
        # LAB VARIABLES (Dynamic)
        self.gamma = 0.9
        self.epsilon = 0.1
        self.penalty = -0.04
        self.slippery = False
        self.alpha = 0.5 

    def reset_agent(self):
        self.agent_pos = self.start_pos
        self.is_game_over = False

    def reset_brain(self):
        self.Q = np.zeros((self.size, self.size, 4))
        self.reset_agent()

    def step(self):
        if self.is_game_over:
            self.reset_agent()
            return

        r, c = self.agent_pos

        # 1. Action Selection (Epsilon-Greedy)
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0,1,2,3])
        else:
            # Add noise to break ties
            action = np.argmax(self.Q[r, c, :] + np.random.randn(4)*0.001)

        # 2. Slippery Logic (Stochasticity)
        move_action = action
        if self.slippery:
            roll = np.random.rand()
            if roll > 0.8: # 20% Chance to slip
                # Slip to perpendicular direction
                if action in [0, 1]: move_action = np.random.choice([2, 3])
                else: move_action = np.random.choice([0, 1])

        # 3. Calculate Move
        dr, dc = 0, 0
        if move_action == 0: dr = -1   # UP
        elif move_action == 1: dr = 1  # DOWN
        elif move_action == 2: dc = -1 # LEFT
        elif move_action == 3: dc = 1  # RIGHT
        
        nr, nc = r + dr, c + dc
        
        # 4. Wall/Boundaries
        if not (0 <= nr < self.size and 0 <= nc < self.size) or (nr, nc) in self.walls:
            nr, nc = r, c 
        
        # 5. Rewards
        reward = self.penalty
        done = False
        
        if (nr, nc) == self.goal_pos:
            reward = 1.0
            done = True
        elif (nr, nc) in self.pits:
            reward = -1.0
            done = True
            
        # 6. Bellman Update (Using Dynamic Gamma)
        old_val = self.Q[r, c, action]
        next_max = 0 if done else np.max(self.Q[nr, nc, :])
        target = reward + self.gamma * next_max
        self.Q[r, c, action] = old_val + self.alpha * (target - old_val)
        
        self.agent_pos = (nr, nc)
        if done: self.is_game_over = True

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
# 4. UI VISUALIZATION (Triangles + Text)
# ==========================================
class TriangleCell(QGraphicsPolygonItem):
    def __init__(self, polygon, parent_scene):
        super().__init__(polygon)
        self.setPen(QPen(C_LINE, 1))
        self.setBrush(QBrush(C_CELL_ZERO))
        parent_scene.addItem(self)

class GridCell:
    def __init__(self, r, c, scene):
        self.triangles = {}
        self.texts = {}
        
        x, y = c * CELL_SIZE, r * CELL_SIZE
        center = QPointF(x + CELL_SIZE/2, y + CELL_SIZE/2)
        tl, tr = QPointF(x, y), QPointF(x + CELL_SIZE, y)
        bl, br = QPointF(x, y + CELL_SIZE), QPointF(x + CELL_SIZE, y + CELL_SIZE)
        
        # Geometries
        coords = {
            0: [tl, tr, center],   # UP
            1: [bl, br, center],   # DOWN
            2: [tl, bl, center],   # LEFT
            3: [tr, br, center]    # RIGHT
        }
        
        # Text Positions (Diamond Layout)
        t_pos = {
            0: (x + CELL_SIZE/2 - 12, y + 5),
            1: (x + CELL_SIZE/2 - 12, y + CELL_SIZE - 20),
            2: (x + 5, y + CELL_SIZE/2 - 10),
            3: (x + CELL_SIZE - 30, y + CELL_SIZE/2 - 10)
        }

        for action in [0,1,2,3]:
            # Draw Triangle
            poly = QPolygonF(coords[action])
            tri = TriangleCell(poly, scene)
            self.triangles[action] = tri
            
            # Draw Text
            txt = QGraphicsTextItem("")
            txt.setFont(QFont("Menlo", 9, QFont.Weight.Bold))
            txt.setDefaultTextColor(C_TEXT)
            txt.setPos(*t_pos[action])
            txt.setZValue(5) # Ensure text is ABOVE color
            scene.addItem(txt)
            self.texts[action] = txt

    def update(self, q_values):
        for action in [0,1,2,3]:
            val = q_values[action]
            
            # 1. Update Color (Gradient)
            self.triangles[action].setBrush(QBrush(get_q_color(val)))
            
            # 2. Update Text (Hide zeros)
            if abs(val) < 0.01:
                self.texts[action].setPlainText("")
            else:
                self.texts[action].setPlainText(f"{val:.2f}")
                # Optional: Make text brighter if background is dark
                self.texts[action].setDefaultTextColor(QColor("white"))

    def set_static(self, color, label, scene):
        """Overlay for Goal/Wall/Pit"""
        x = self.triangles[0].polygon().at(0).x()
        y = self.triangles[0].polygon().at(0).y()
        rect = scene.addRect(x, y, CELL_SIZE, CELL_SIZE)
        
        if label == "WALL":
            rect.setBrush(QBrush(color, Qt.BrushStyle.DiagCrossPattern))
        else:
            rect.setBrush(QBrush(color))
        rect.setPen(QPen(C_LINE, 2))
        
        if label:
            t = scene.addText(label)
            t.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            t.setDefaultTextColor(C_BG_DARK)
            br = t.boundingRect()
            t.setPos(x + (CELL_SIZE - br.width())/2, y + (CELL_SIZE - br.height())/2)

# ==========================================
# 4. MAIN APP (Window + Controls)
# ==========================================
class MarsRoverLab(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mars Rover RL: Final Lab")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"QMainWindow {{ background-color: {C_BG_DARK.name()}; }} QLabel {{ color: {C_TEXT.name()}; }}")
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # 1. GRAPHICS VIEW
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(C_BG_DARK))
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 1px solid {C_LINE.name()}; border-radius: 4px;")
        layout.addWidget(self.view)
        
        self.engine = MarsRoverEngine()
        self.cells = {}
        self.init_graphics()
        
        # 2. CONTROL DECK
        self.create_control_deck(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)

    def create_control_deck(self, parent_layout):
        deck = QFrame()
        deck.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 10px; margin-top: 10px;")
        deck_layout = QVBoxLayout(deck)
        
        # --- SLIDERS ---
        param_group = QGroupBox("Lab Parameters")
        param_group.setStyleSheet(f"QGroupBox {{ color: {C_POS_LOW.name()}; font-weight: bold; border: 1px solid {C_LINE.name()}; margin-top: 15px; }} QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }}")
        grid = QGridLayout(param_group)
        
        # Fixed Slider Helper
        def add_slider(label, min_v, max_v, init_v, r, c, logic_handler):
            # logic_handler: Function that takes (int) -> updates engine -> returns (str)
            
            # 1. Get initial formatted text (e.g., "0.90") using the handler
            initial_text = logic_handler(init_v)
            l = QLabel(f"{label}: {initial_text}")
            
            s = QSlider(Qt.Orientation.Horizontal)
            s.setMinimum(min_v)
            s.setMaximum(max_v)
            s.setValue(int(init_v))
            
            # 2. Connect update function
            def on_change(v):
                text_val = logic_handler(v)
                l.setText(f"{label}: {text_val}")
                
            s.valueChanged.connect(on_change)
            grid.addWidget(l, r, c)
            grid.addWidget(s, r+1, c)
            return s

        # --- DEFINING THE LOGIC HANDLERS ---
        
        # Gamma: Slider 90 -> Float 0.90
        def handle_gamma(v):
            val = v / 100.0
            self.engine.gamma = val
            return f"{val:.2f}"

        # Epsilon: Slider 10 -> Float 0.10
        def handle_eps(v):
            val = v / 100.0
            self.engine.epsilon = val
            return f"{val:.2f}"

        # Penalty: Slider 4 -> Float -0.04
        def handle_penalty(v):
            val = -v / 100.0
            self.engine.penalty = val
            return f"{val:.2f}"

        # --- ADDING THE SLIDERS ---
        add_slider("Gamma (γ)", 10, 99, 90, 0, 0, handle_gamma)
        add_slider("Exploration (ε)", 0, 100, 10, 0, 1, handle_eps)
        add_slider("Living Cost", 0, 200, 4, 0, 2, handle_penalty)

        # Checkbox
        chk = QCheckBox("Slippery Floor")
        chk.setStyleSheet(f"color: {C_TEXT.name()}; font-weight: bold;")
        chk.toggled.connect(lambda c: setattr(self.engine, 'slippery', c))
        grid.addWidget(chk, 1, 3)

        deck_layout.addWidget(param_group)

        # --- BUTTONS ---
        btn_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("▶ Run Experiment")
        self.btn_play.setCheckable(True)
        self.btn_play.setStyleSheet(f"background-color: {C_POS_LOW.name()}; color: white; border-radius: 5px; padding: 8px;")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_layout.addWidget(self.btn_play)
        
        self.btn_reset = QPushButton("Reset Brain")
        self.btn_reset.setStyleSheet(f"background-color: {C_NEG_HIGH.name()}; color: white; border-radius: 5px; padding: 8px;")
        self.btn_reset.clicked.connect(self.reset)
        btn_layout.addWidget(self.btn_reset)

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

    def tick(self):
        self.engine.step()
        self.update_agent()
        for (r,c), cell in self.cells.items():
            cell.update(self.engine.Q[r,c])

    def toggle_play(self):
        if self.btn_play.isChecked():
            self.btn_play.setText("⏸ Pause")
            self.timer.start(25)
        else:
            self.btn_play.setText("▶ Run Experiment")
            self.timer.stop()

    def reset(self):
        self.engine.reset_brain()
        for cell in self.cells.values(): cell.update(np.zeros(4))
        self.update_agent()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MarsRoverLab()
    win.show()
    sys.exit(app.exec())