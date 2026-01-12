# AI Anaesthesiologist: Personalised Drug Dosing with PPO

This project implements a **Reinforcement Learning (RL)** agent capable of controlling the delivery of Propofol anaesthesia. Using **Proximal Policy Optimisation (PPO)** and the **Schnider Pharmacokinetic/Pharmacodynamic (PK/PD)** model, the agent learns to maintain patients at a target Bispectral Index (BIS) depth of 50. For further information read [my medium page](https://medium.com/@tian.pan/an-ai-anaesthesiologist-using-ppo-for-safe-personalised-drug-dosing-23b9ad910bb7).

> **⚠️ Medical Disclaimer:** This project is a proof-of-concept conducted entirely in a simulation environment. The PK/PD parameters are mathematical approximations and **must not** be used for clinical decision-making or real-world drug delivery.


---

## Key Features

* **Context-Aware Dosing:** The agent observes demographic data (Age, Gender, Weight, Height) and adapts its infusion strategy accordingly.
* **Schnider PK/PD Simulation:** Implements a realistic physiological model for Propofol distribution (Blood → Muscle → Fat → Brain).
* **Safety-First Reward Function:** Prioritises patient safety with heavy penalties for "overdose" states (BIS < 40).
* **Stable Baselines 3 Implementation:** Utilises industry-standard PPO algorithms for stable, clipped policy updates.

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Tian807/PPO_Anaesthesia.git
cd PPO_Anaesthesia

```


2. **Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install gymnasium numpy matplotlib stable-baselines3 pandas

```



---

## 📂 Code Structure

* `patient_env.py`: The custom Gym environment. Contains the **Schnider PK/PD equations**, the `step` logic, and the reward function.
* `train_agent.py`: Script to initialise the PPO agent and train it using Stable Baselines 3.
* `generate_plots.py`: Loads the trained model and runs simulations on unseen patients (Young vs Elderly, Male vs Female).
* `ppo_anaesthesia_agent.zip`: The saved weights of the trained model.

---

## 🧠 The Environment

The agent interacts with a custom `gymnasium` environment representing a virtual patient.

### Observation Space (7 Variables)

The agent receives a vector containing:

1. **Current BIS Score:** (0 = Coma, 100 = Awake)
2. **Target BIS:** (Fixed at 50)
3. **Effect Site Concentration:** Estimated drug in the brain.
4. **Demographics:** [Age, Weight, Height, Gender]

### Action Space (Continuous)

* **Propofol Infusion Rate:** A continuous value mapped to **0 – 10 mg/sec**.

### Reward Function

The agent optimises for:

$$R = 0.5 - |Error| - (0.1 \times Dose) - (5.0 \text{ if } BIS < 40)$$

This structure encourages accuracy while strictly penalising dangerous overdoses.

---

## 📝 Usage

**To train a new agent:**

```bash
python train_agent.py

```

**To evaluate the pre-trained model:**

```bash
python generate_plots.py

```

**To continue training the pre-trained model:**
```bash
python continue_training.py

```
