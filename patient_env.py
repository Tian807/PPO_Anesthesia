import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AnaesthesiaEnv(gym.Env):
    def __init__(self):
        super(AnaesthesiaEnv, self).__init__()
        
        # Action: Propofol infusion rate (Normalised for PPO: -1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: [Current BIS, Target BIS, Effect Concentration, Age, Weight, Height, Gender]
        self.observation_space = spaces.Box(low=0, high=200, shape=(7,), dtype=np.float32)
        
        self.target_bis = 50.0
        self.dt = 1.0  # Time step: 1 second
        
        # PD parameters (Hill Equation - Sigmoid Curve)
        self.E0 = 100.0
        self.Emax = 100.0
        self.EC50 = 4.0
        self.gamma = 2.0

    def _get_schnider_params(self, age, weight, height, gender):
        """
        Calculates PK constants based on the Schnider Model.
        This enables the agent to treat random patients (Domain Randomisation).
        Values taken from: Schnider, T. W., et al. (1998). "The influence of age on propofol pharmacodynamics."
        """
        # Lean Body Mass (lbm) calculated from James Equation
        if gender == 0: # Male
            lbm = 1.1 * weight - 128 * (weight / height) ** 2
        else: # Female
            lbm = 1.07 * weight - 148 * (weight / height) ** 2
        
        # Schnider Model Equations for Compartment Volumes (V) and Transfer Rates (k)
        v1 = 4.27
        v2 = 18.9 - 0.391 * (age - 53)
        v3 = 238

        k10 = 0.443 + 0.0107 * (weight - 77) - 0.0159 * (lbm - 59) + 0.0062 * (height - 177)
        k12 = 0.302 - 0.0056 * (age - 53)
        k13 = 0.196
        k21 = (1.29 - 0.024 * (age - 53)) / v2
        k31 = 0.0035
        ke0 = 0.456

        return k10, k12, k13, k21, k31, ke0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Generate a random patient
        if options and 'age' in options:
            self.age = options['age']
            self.weight = options['weight']
            self.height = options['height']
            self.gender = options['gender']
        else:
            self.age = np.random.randint(20, 85)
            self.weight = np.random.randint(50, 110)
            self.height = np.random.randint(150, 195)
            self.gender = np.random.choice([0, 1])
        
        # 2. Update Physics Engine for this patient using Schnider Model
        self.k10, self.k12, self.k13, self.k21, self.k31, self.ke0 = \
            self._get_schnider_params(self.age, self.weight, self.height, self.gender)
            
        # 3. Reset State [Central, Rapid, Slow, Effect]
        self.state = np.zeros(4) 
        self.current_bis = 100.0
        self.steps = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Normalise demographics (divide by 100) so Neural Network learns faster
        return np.array([
            self.current_bis, 
            self.target_bis, 
            self.state[3], # Effect site concentration
            self.age / 100.0, 
            self.weight / 100.0, 
            self.height / 100.0, 
            self.gender
        ], dtype=np.float32)

    def step(self, action):
        # 1. Convert Action: PPO outputs [-1, 1], we map to [0, 10] mg/sec
        infusion_rate = float(np.clip((action[0] + 1) * 5, 0, 10))
        
        # 2. Physics Step (Pharmacokinetics - Euler Integration) 
        x1, x2, x3, xe = self.state
        dx1 = infusion_rate + self.k21 * x2 + self.k31 * x3 - (self.k10 + self.k12 + self.k13) * x1
        dx2 = self.k12 * x1 - self.k21 * x2
        dx3 = self.k13 * x1 - self.k31 * x3
        dxe = self.ke0 * (x1 - xe)
        self.state += np.array([dx1, dx2, dx3, dxe]) * self.dt
        
        # 3. Calculate BIS (Pharmacodynamics - Hill Equation)
        Ce = self.state[3]
        effect = (self.Emax * (Ce**self.gamma)) / (Ce**self.gamma + self.EC50**self.gamma)
        self.current_bis = self.E0 - effect
        
        # 4. REWARD FUNCTION (Implementation of Schamberg et al., Eq. 4)
        # r = 0.5 - |error| - (rho1 * dose) - (rho2 * safety_violation)
        
        # Calculate normalized error (Target - Current)
        error = (self.target_bis - self.current_bis) / 50.0 
        
        # Base Reward (Reward for keeping patient alive)
        reward = 0.5
        
        # A. Performance Penalty (minimise error)
        reward -= abs(error)
        
        # B. Sparsity Penalty (Rho 1) - "Use less drug"
        reward -= 0.1 * infusion_rate 
        
        # C. Safety Penalty (Rho 2) - "Avoid over-sedation"
        # If BIS drops below 40, harsh penalty
        if self.current_bis < 40:
            reward -= 5.0 

        self.steps += 1
        truncated = self.steps >= 300 # Episode ends after 5 minutes (300 secs)
        
        return self._get_obs(), reward, False, truncated, {"bis": self.current_bis, "infusion": infusion_rate}