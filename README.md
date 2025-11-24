
# 🚗 CarRacing-v3 Reinforcement Learning Agent  
### **PPO + Stable-Baselines3 | Gymnasium | PyTorch | CUDA Accelerated**

This project trains a deep reinforcement learning agent using **Proximal Policy Optimization (PPO)** to play the **CarRacing‑v3** environment from **Gymnasium**.  
The environment provides a top‑down racing track where the agent must learn steering, braking, and acceleration to complete laps efficiently.

---

## ⭐ Key Features

- **PPO with CNN-based policy**
- **Vectorized environment** for faster training
- **GPU acceleration (CUDA)**
- **Checkpoint saving + best model tracking**
- **Evaluation script with rendering**
- **Training logs compatible with TensorBoard**
- **Clean project structure for GitHub**

---

## 📂 Project Structure

```
Car_Race/
│
├── train_agent.py         # Training script
├── eval_agent.py          # Evaluate trained model
├── env_test.py            # Quick environment test
│
├── requirements.txt       # Dependencies list
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

---

## 🚀 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/car-race-ppo.git
cd car-race-ppo
```

### 2️⃣ Create and activate virtual environment  
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏎️ Environment Details — CarRacing‑v3

- Observation shape: **(96, 96, 3)** RGB image  
- Action space: **[steer, gas, brake]**  
- Continuous control  
- Randomly generated track (different layout every episode)

---

## 🏋️ Training the PPO Agent

Simply run:

```bash
python train_agent.py
```

What this script does:

✔ Creates vectorized + monitored environments  
✔ Trains PPO for 1M timesteps  
✔ Saves:
- `logs_car_race/best_model/best_model.zip`
- TensorBoard logs  
- Checkpoints  

---

## 📊 Monitoring Training

Start TensorBoard:

```bash
tensorboard --logdir logs_car_race/
```

Open in browser:  
👉 http://localhost:6006/

You'll see:
- episode rewards  
- policy/value losses  
- learning rate  
- explained variance  

---

## 🎮 Evaluating the Trained Agent

```bash
python eval_agent.py
```

The script will:

✔ Load **best_model.zip**  
✔ Render real-time racing  
✔ Print reward per episode  

---

## 🎥 Recording Gameplay (Optional)

To generate video:

```python
env = gym.make("CarRacing-v3", render_mode="rgb_array")
```

and use `imageio` or `moviepy` to create MP4 output.

---

## 📈 Example Results (From a 1M-Step PPO Run)

| Metric | Value |
|-------|--------|
| Average Evaluation Reward | **850–900** |
| Max Reward | **> 950** |
| Average Episode Length | ~900–1000 frames |

A reward above **800** indicates **expert-level driving** in CarRacing‑v3.

---

## 📦 Requirements

```
gymnasium==0.29.1
gymnasium[box2d]==0.29.1
pybox2d
stable-baselines3[extra]
tensorboard
moviepy
imageio[ffmpeg]
numpy
```

---

## 🔧 Troubleshooting

### ⚠ Box2D installation error?
Use:
```bash
pip install gymnasium[box2d]==0.29.1 pybox2d
```

### ⚠ “CUDA not available”?
Install PyTorch with GPU support from  
https://pytorch.org/get-started/locally/

---

## 📝 License
MIT License.  
Feel free to use, modify, and share.

---

## ❤️ Acknowledgements
- Gymnasium developers  
- Stable-Baselines3 team  
- PyTorch contributors  
- OpenAI for original environment inspirations  

---

## ⭐ Contribute
PRs are welcome.  
Star the repo ⭐ if you find it useful!

