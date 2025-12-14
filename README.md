# 🚗 NeuralNav: Self-Driving Car with Deep Q-Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyQt6-6.0+-green.svg" alt="PyQt6">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <a href="https://youtu.be/giIbDXvCTaI"><img src="https://img.shields.io/badge/YouTube-Demo-red?logo=youtube" alt="YouTube Demo"></a>
</p>

An interactive **Deep Q-Network (DQN)** reinforcement learning simulation where an AI-powered car learns to navigate city maps, avoid obstacles, and reach multiple waypoints autonomously.

---

## 🎬 Demo Video

<p align="center">
  <a href="https://youtu.be/giIbDXvCTaI">
    <img src="https://img.youtube.com/vi/giIbDXvCTaI/maxresdefault.jpg" alt="Watch Demo Video" width="600">
  </a>
</p>

<p align="center">
  <a href="https://youtu.be/giIbDXvCTaI">▶️ Watch the Full Demo on YouTube</a>
</p>

---

## 🗺️ Sample Map

![Demo Preview](radial-map.png)

---

## ✨ Features

- 🧠 **Deep Q-Network (DQN)** with target network and soft updates (Polyak averaging)
- 🎯 **Multi-Waypoint Navigation** - Set multiple targets for sequential navigation
- 📊 **Real-Time Visualization** - Watch the car learn with live reward charts
- 🔍 **7 Sensor System** - Car detects obstacles using simulated distance sensors
- 💾 **Prioritized Experience Replay** - Faster learning by prioritizing successful episodes
- 🎨 **Nordic-Themed GUI** - Beautiful, modern interface built with PyQt6
- 🗺️ **Custom Maps** - Load any image as a navigation map
- ⚙️ **Educational Assignment Mode** - Learn RL by fixing intentionally broken parameters

---

## 🏗️ Architecture

### Neural Network
```
Input (9) → Linear(128) → ReLU → Linear(256) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → Output (5 actions)
```

### State Space (9 dimensions)
| Index | Description |
|-------|-------------|
| 0-6 | 7 sensor readings (obstacle detection at -45°, -30°, -15°, 0°, 15°, 30°, 45°) |
| 7 | Normalized angle to target |
| 8 | Normalized distance to target |

### Action Space (5 actions)
| Action | Description |
|--------|-------------|
| 0 | Turn Left |
| 1 | Go Straight |
| 2 | Turn Right |
| 3 | Sharp Left Turn |
| 4 | Sharp Right Turn |

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/rl-bellman-equation.git
cd rl-bellman-equation

# Install dependencies
pip install torch numpy PyQt6
```

### Run the Simulation
```bash
python citymap_assignment.py
```

---

## 🎮 How to Use

1. **Launch the application** - A window will open with the city map
2. **Set Car Position** - Click on the map to place the car's starting position
3. **Set Target(s)** - Left-click to add waypoints (multiple clicks for sequential targets)
4. **Finish Setup** - Right-click when done adding targets
5. **Start Training** - Press `SPACE` or click the START button
6. **Watch & Learn** - Observe the car learning in real-time!

### Controls
| Key/Action | Function |
|------------|----------|
| Left Click (setup) | Place car, then add targets |
| Right Click | Finish target setup |
| Space | Start/Pause training |
| Load Map | Import custom map images |
| Reset | Clear all and start over |

---

## 📚 Key Concepts

### Bellman Equation
The core of Q-Learning, used to update Q-values:

$$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

Where:
- `Q(s, a)` = Expected future reward for taking action `a` in state `s`
- `r` = Immediate reward
- `γ` (gamma) = Discount factor for future rewards
- `s'` = Next state

### DQN Improvements Used
- **Experience Replay** - Store and sample past experiences randomly
- **Prioritized Replay** - Sample successful episodes more frequently
- **Target Network** - Separate network for stable Q-value targets
- **Soft Updates** - Gradual target network updates using Polyak averaging

---

## 📐 Bellman Equation Demo (`bellman_equation.py`)

A standalone script demonstrating **iterative policy evaluation** using the Bellman equation on a 4×4 grid world.

### Grid World Setup
- **4×4 grid** with terminal state at bottom-right corner (3,3)
- **Actions**: North, South, East, West (equal probability: 0.25 each)
- **Rewards**: -1 for each step, 0 at terminal state
- **Discount factor (γ)**: 1.0

### How It Works

The script iteratively updates the value function using:

$$V(s) = \sum_{a} \pi(a|s) \left[ R(s,a) + \gamma V(s') \right]$$

### Run the Demo
```bash
python bellman_equation.py
```

### Output (Converged Value Function)

After iterating until convergence (θ = 1e-5), the value function converges to:

```
=======================
-58.4281 -56.4281 -53.2853 -50.7139
-56.4281 -53.5710 -48.7139 -44.1425
-53.2853 -48.7139 -39.8568 -28.9998
-50.7139 -44.1425 -28.9998  0.0000
=======================
```

The values represent the **expected cumulative reward** from each state under a random policy. Notice how states closer to the terminal (bottom-right) have higher values (less negative).

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 256 | Experiences per training step |
| `GAMMA` | 0.9 | Discount factor (0-1) |
| `LR` | 0.0001 | Learning rate |
| `TAU` | 0.001 | Soft update coefficient |
| `EPSILON` | 1.0 → 0.001 | Exploration rate (decays) |
| `SENSOR_DIST` | 10 px | Sensor detection range |
| `SPEED` | 5 px/step | Car forward speed |

---

## 📁 Project Structure

```
rl-bellman-equation/
├── citymap_assignment.py   # Main DQN car navigation (assignment version)
├── bellman_equation.py     # Bellman equation demonstrations
├── bellman-equation.ipynb  # Jupyter notebook tutorial
├── city_map.png            # Default navigation map
├── radial-map.png          # Alternative map
├── ClassCode/              # Additional learning materials
│   ├── mars_rover_dqn.py   # Mars rover DQN example
│   ├── mars_rover_mdp.py   # Markov Decision Process demo
│   ├── mars_rover_ui.py    # Mars rover UI
│   └── mars_rover.py       # Mars rover base implementation
└── README.md
```

---

## 🎓 Assignment Mode

This project includes an **educational assignment** where key parameters are intentionally set to incorrect values. Students must:

1. Find parameters marked with `# FIX ME!`
2. Understand what each parameter does
3. Set appropriate values based on RL knowledge
4. Test and observe learning improvements

### Parameters to Fix:
- `SENSOR_DIST` - Sensor detection distance
- `SENSOR_ANGLE` - Sensor spread angle
- `SPEED` - Car forward speed
- `TURN_SPEED` - Turn angle per step
- `SHARP_TURN` - Sharp turn angle
- `BATCH_SIZE` - Training batch size
- `GAMMA` - Discount factor
- `LR` - Learning rate
- `TAU` - Soft update coefficient
- `epsilon` - Initial exploration rate

---

## 🔧 Customization

### Loading Custom Maps
1. Click **"📂 LOAD MAP"** in the GUI
2. Select any PNG/JPG image
3. **White pixels** = Drivable road
4. **Dark pixels** = Obstacles/walls

### Creating Your Own Maps
- Use any image editor (Photoshop, GIMP, Paint)
- Draw roads in white (#FFFFFF)
- Draw obstacles in black or dark colors
- Save as PNG or JPG

---

## 📈 Reward System

| Event | Reward |
|-------|--------|
| Crash (hit wall) | -100 |
| Reach target | +100 |
| Stay on road (center) | +0 to +20 |
| Move away from target | -10 |
| Each step | -0.1 |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs
- 💡 Suggest features
- 🔧 Submit pull requests

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch team for the amazing deep learning framework
- Qt/PyQt6 for the powerful GUI toolkit
- The RL research community for DQN innovations

---
