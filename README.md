# 🚗 NeuralNav: Self-Driving Car with Deep Q-Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyQt6-6.0+-green.svg" alt="PyQt6">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

An interactive **Deep Q-Network (DQN)** reinforcement learning simulation where an AI-powered car learns to navigate city maps, avoid obstacles, and reach multiple waypoints autonomously.

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
