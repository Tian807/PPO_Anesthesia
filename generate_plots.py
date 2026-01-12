import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from patient_env import AnaesthesiaEnv

# --- SETUP ---
# Load the environment and your trained model
env = AnaesthesiaEnv()
model = PPO.load("ppo_anaesthesia_agent")
plt.style.use('seaborn-v0_8-whitegrid') 

# ========================================================
#  Test 1: YOUNG vs. ELDERLY PATIENT
# ---------------------------------------------------------
print("Running Clinical Trials: Young vs. Elderly...")

def run_clinical_trial(age, weight, height, gender):
    # Force the environment to create a specific patient
    obs, _ = env.reset(options={'age': age, 'weight': weight, 'height': height, 'gender': gender})
    
    bis_history = []
    dose_history = []
    
    for _ in range(1800):
        # Predict action (Deterministic = remove uncertainty for testing)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, truncated, info = env.step(action)
        
        bis_history.append(info['bis'])
        dose_history.append(info['infusion'])
        
        if truncated: break
    return bis_history, dose_history

# Patient A: 25-year-old Male 
young_bis, young_dose = run_clinical_trial(age=25, weight=70, height=180, gender=0)

# Patient B: 80-year-old Male 
old_bis, old_dose = run_clinical_trial(age=80, weight=70, height=180, gender=0)

# Plotting the Comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Graph 1: BIS Response
ax1.plot(young_bis, label="Young Patient (25y)", color='blue', linewidth=2)
ax1.plot(old_bis, label="Elderly Patient (80y)", color='orange', linewidth=2)
ax1.axhline(y=50, color='green', linestyle=':', label="Target BIS (50)")
ax1.set_ylabel("BIS Score")
ax1.set_title("PPO Performance: Young vs. Elderly Patient")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graph 2: Dosing Strategy
ax2.plot(young_dose, label="Dose for Young", color='blue', alpha=0.6)
ax2.plot(old_dose, label="Dose for Elderly", color='orange', alpha=0.8)
ax2.set_ylabel("Propofol Infusion (mg/sec)")
ax2.set_xlabel("Time (s)")
ax2.set_title("Context-Aware Dosing: The Agent 'Learned' Safety")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("article_comparison_graph.png")
print("Graph saved! Check your folder for 'article_comparison_graph.png'")
plt.show()

# ---------------------------------------------------------
# TEST. 2: GENDER COMPARISON 
# ---------------------------------------------------------
print("Running Gender Comparison Test...")

# Control Variables (Keep everything else constant)
test_age = 40
test_weight = 70
test_height = 170

# Patient C: Male
male_bis, male_dose = run_clinical_trial(age=test_age, weight=test_weight, height=test_height, gender=0)

# Patient D: Female
female_bis, female_dose = run_clinical_trial(age=test_age, weight=test_weight, height=test_height, gender=1)

# Plotting the Gender Difference
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Graph 1: BIS Response
ax1.plot(male_bis, label="Male Patient", color='blue', linewidth=2)
ax1.plot(female_bis, label="Female Patient", color='magenta', linestyle='--', linewidth=2)
ax1.axhline(y=50, color='green', linestyle=':', label="Target")
ax1.set_title("Pharmacokinetic Difference: Male vs Female (Same Weight/Height)")
ax1.set_ylabel("BIS Score")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graph 2: Dosing
ax2.plot(male_dose, label="Male Dose", color='blue', alpha=0.6)
ax2.plot(female_dose, label="Female Dose", color='magenta', alpha=0.6)
ax2.set_title("AI Dosing Adjustment based on Gender")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Infusion Rate (mg/sec)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gender_comparison.png")
plt.show()

# ========================================================
# PLOT 3: POLICY HEATMAP
# ========================================================
print("Generating Policy Heatmap...")

# 1. Create a grid of states
bis_values = np.linspace(20, 80, 50)  # X-axis: BIS from Deep (20) to Awake (80)
conc_values = np.linspace(0, 6, 50)   # Y-axis: Brain Concentration 0 to 6
action_grid = np.zeros((50, 50))

# 2. Ask the agent what it would do in each state
# We fix the patient to a standard 40yr old male for this test
fixed_demographics = [0.40, 0.70, 1.70, 0.0] # Age 40, 70kg, 170cm, Male

for i, conc in enumerate(conc_values):
    for j, bis in enumerate(bis_values):
        # Construct observation: [BIS, Target, Conc, Age, Weight, Height, Gender]
        obs = np.array([bis, 50.0, conc] + fixed_demographics, dtype=np.float32)
        
        # Predict action (Deterministic)
        action, _ = model.predict(obs, deterministic=True)
        
        # Convert PPO action (-1 to 1) to mg/sec (0 to 10)
        infusion_mg = np.clip((action[0] + 1) * 5, 0, 10)
        action_grid[i, j] = infusion_mg

# 3. Plot Heatmap
plt.figure(figsize=(10, 6))
# Flip Y-axis so 0 is at bottom
sns.heatmap(action_grid[::-1], xticklabels=5, yticklabels=5, cmap="viridis", 
            cbar_kws={'label': 'Propofol Dose (mg/sec)'})

plt.title("AI Policy Map: Decision Making Surface", fontsize=14, fontweight='bold')
plt.xlabel("Current BIS Value (Patient Consciousness)", fontsize=12)
plt.ylabel("Current Effect Site Concentration", fontsize=12)

# Fix axis labels to match ranges
plt.xticks(np.linspace(0, 49, 5), labels=[20, 35, 50, 65, 80])
plt.yticks(np.linspace(0, 49, 5), labels=[6.0, 4.5, 3.0, 1.5, 0.0])

plt.tight_layout()
plt.savefig("plot_policy_heatmap.png")
print("Saved 'plot_policy_heatmap.png'")


# ========================================================
# PLOT 3: POPULATION ROBUSTNESS (100 Random Patients)
# ========================================================
print("\nRunning Population Analysis (100 Patients)...")

ttr_scores = [] # Time in Target Range (BIS 40-60)

for i in range(100):
    obs, _ = env.reset() # Random patient every time
    steps_in_target = 0
    total_steps = 300
    
    for _ in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, truncated, info = env.step(action)
        
        # Check if patient is in safe zone (40-60)
        if 40 <= info['bis'] <= 60:
            steps_in_target += 1
        
        if truncated: break
            
    ttr_scores.append((steps_in_target / total_steps) * 100)

plt.figure(figsize=(8, 5))
plt.hist(ttr_scores, bins=15, color='#2ecc71', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(ttr_scores), color='red', linestyle='--', linewidth=2, label=f'Mean Accuracy: {np.mean(ttr_scores):.1f}%')

plt.title("Robustness Analysis: Performance Across 100 Random Patients", fontsize=14)
plt.xlabel("% Time in Target Range (BIS 40-60)", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.savefig("plot_population_robustness_100.png")
print("Saved 'plot_population_robustness_100.png'")

plt.show()