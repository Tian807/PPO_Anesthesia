import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from patient_env import AnaesthesiaEnv  # Assuming your environment file is named patient_env.py

# Callback to log rewards during training
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.rewards = []

    def _on_step(self) -> bool:
        # SB3 wraps environments in a Vector, so 'rewards' is an array. 
        # We take the first one [0] since we have a single env.
        if 'rewards' in self.locals:
            self.rewards.append(self.locals['rewards'][0])
        return True

# Setup and training the PPO agent
env = AnaesthesiaEnv()
print("Starting PPO Training...")

reward_callback = RewardLogger()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# Train the Agent
model.learn(total_timesteps=25, callback=reward_callback)

print("Training Complete!")
model.save("ppo_anaesthesia_agent")
print("Model saved as 'ppo_anaesthesia_agent.zip'")

# Plotting the Learning Curve
print("Generating Learning Curve...")

# Extract rewards from the callback
raw_rewards = np.array(reward_callback.rewards)

# Smooth the data (Moving Average)
# Since we have 150k steps, a larger window (e.g., 1000) makes the plot cleaner
window = 1000 
smoothed_rewards = np.convolve(raw_rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(smoothed_rewards, color='#e74c3c', linewidth=2)

plt.title("Learning Curve", fontsize=14)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Average Reward (Smoothed)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig("plot_learning_curve.png")
print("Saved plot to 'plot_learning_curve.png'")

plt.show()