import gymnasium as gym
import numpy as np
import torch
import random
import copy
import tqdm
import matplotlib.pyplot as plt
import warnings
import argparse
import os
from torch import nn

# Import local utilities
import utils.envs as envs
import utils.seed
import utils.torch
# (Removed unused math.log and utils.buffers)

# Suppress warnings
warnings.filterwarnings("ignore")

# PPO (Proximal Policy Optimization)

def setup_parser() -> argparse.Namespace:
    """Configures and parses command-line arguments."""
    parser = argparse.ArgumentParser()
    # either:
    # cartpole - default cartpole environment
    # mountain_car - default mountain car environment
    # mountain_car_mod - mountain car environment with modified reward
    parser.add_argument('--mode', type=str, default="cartpole")
    return parser.parse_args()


def get_env_config(mode: str) -> dict:
    """Returns environment-specific constants based on the mode."""
    if mode == "cartpole":
        return {
            "OBS_N": 4,           # State space size
            "ACT_N": 2,           # Action space size
            "ENV_NAME": "CartPole-v0",
            "GAMMA": 1.0,         # Discount factor
            "LEARNING_RATE": 5e-4,
            "EPOCHS": 150         # PPO typically converges faster
        }
    elif "mountain_car" in mode:
        return {
            "OBS_N": 2,
            "ACT_N": 3,
            "ENV_NAME": "MountainCar-v0",
            "GAMMA": 0.9,         # Discount factor
            "LEARNING_RATE": 1e-3,
            "EPOCHS": 150         # PPO typically converges faster
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

# --- Global Constants ---
SEED = 1
EPISODES_PER_EPOCH = 1    # Episodes per epoch
TEST_EPISODES = 10        # Test episodes
HIDDEN = 32               # Hidden size
CLIP_EPSILON = 0.2        # PPO clipping parameter

# Setup device
t = utils.torch.TorchHelper()
DEVICE = t.device

# --- Global Network Definitions ---
# (Must be global for the `policy` function to access them)
pi = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(DEVICE) # Actor
V = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(DEVICE)  # Critic


def policy(env, obs: np.ndarray | tuple | dict) -> int:
    """
    Selects an action based on the policy network (actor) given an observation.
    (Note: 'env' param is unused but kept for compatibility with play_episode)
    """
    # Check if obs is a tuple (new gymnasium observation format)
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract the observation array
    # Check if obs is a dictionary
    if isinstance(obs, dict) and 'observation' in obs:
        obs = obs['observation']
    
    obs_tensor = t.f(obs) # Convert state to tensor
    
    # Use Softmax to get action probabilities
    probs = torch.nn.Softmax(dim=-1)(pi(obs_tensor))
    
    # Sample an action from the probability distribution
    return np.random.choice(int(probs.shape[0]), p=probs.cpu().detach().numpy())


def train(S: torch.Tensor, A: torch.Tensor, returns: torch.Tensor, old_log_probs: torch.Tensor,
          OPT: torch.optim.Optimizer, V_OPT: torch.optim.Optimizer,
          PPO_EPOCHS: int, CLIP_EPSILON: float):
    """
    Performs PPO optimization over a batch of data for PPO_EPOCHS.
    (This function uses the global `pi` and `V` networks)
    """
    # PPO uses multiple epochs of optimization over the collected batch
    for _ in range(PPO_EPOCHS):
        OPT.zero_grad()
        V_OPT.zero_grad()
        
        # --- Critic (Value Network) Update ---
        # 1. Calculate current state values: V(s_t)
        state_values = V(S).squeeze()
        # 2. Calculate Value loss: MSE(G_t, V(s_t))
        value_loss = torch.nn.functional.mse_loss(state_values, returns)
        # 3. Update Critic
        value_loss.backward()
        V_OPT.step()

        # --- Actor (Policy Network) Update ---
        # 1. Calculate advantage: A_t = G_t - V(s_t)
        # We detach state_values: gradients from policy loss should not flow to the critic
        advantages = (returns - state_values).detach()
        
        # 2. Calculate current log probabilities: log(pi_new(a_t | s_t))
        current_log_probs = torch.nn.LogSoftmax(dim=-1)(pi(S)).gather(1, A.view(-1, 1)).view(-1)
        
        # 3. Calculate the ratio: r_t = exp(log_prob_new - log_prob_old)
        # Detach old_log_probs as they are constants in this optimization
        ratio = torch.exp(current_log_probs - old_log_probs.detach())
        
        # 4. Calculate PPO clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
        
        # 5. Policy loss (Actor loss) - take the minimum and mean
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 6. Update Actor
        policy_loss.backward()
        OPT.step()


def main():
    """Main execution block for training and evaluation."""
    global pi, V # Declare intent to modify the global policy/value networks
    
    args = setup_parser()
    
    # --- 1. Setup Configuration ---
    config = get_env_config(args.mode)
    
    OBS_N = config["OBS_N"]
    ACT_N = config["ACT_N"]
    ENV_NAME = config["ENV_NAME"]
    GAMMA = config["GAMMA"]
    LEARNING_RATE = config["LEARNING_RATE"]
    EPOCHS = config["EPOCHS"]
    
    # PPO specific parameters
    PPO_EPOCHS = 10 # Number of PPO optimization epochs over the collected batch
    
    # --- 2. Initialize Environment and Seed ---
    utils.seed.seed(SEED)
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    # Seed the environment during the first reset
    env.reset(seed=SEED) 
    
    # --- 3. Initialize Networks and Optimizers ---
    # (Re-initialize the global networks with correct dimensions)
    pi = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    
    V = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1) # Output is a single value
    ).to(DEVICE)
    
    OPT = torch.optim.Adam(pi.parameters(), lr=LEARNING_RATE)
    V_OPT = torch.optim.Adam(V.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    Rs = []
    last25Rs = []
    print("Training:")
    pbar = tqdm.trange(EPOCHS)
    for epi in pbar:

        all_S, all_A, all_old_log_probs = [], [], []
        all_returns = []

        for _ in range(EPISODES_PER_EPOCH):
            
            # Play an episode
            S, A, R = envs.play_episode(env, policy)

            # Get log probs of actions taken under the *current* policy
            # (these will be the "old" log probs for the update)
            with torch.no_grad():
                S_tensor_for_log_probs = t.f(S[:-1])
                A_tensor_for_log_probs = t.l(np.array(A))
                old_log_probs_episode = torch.nn.LogSoftmax(dim=-1)(pi(S_tensor_for_log_probs))\
                                              .gather(1, A_tensor_for_log_probs.view(-1, 1))\
                                              .view(-1)

            # Modify reward for "mountain_car_mod"
            if args.mode == "mountain_car_mod":
                R = [s[0] for s in S[:-1]]

            all_S += S[:-1]  # ignore last state
            all_A += A
            all_old_log_probs.append(old_log_probs_episode)
            
            # Create returns (G_t = "return-to-go")
            discounted_rewards = copy.deepcopy(R)
            for i in range(len(R) - 1)[::-1]:
                discounted_rewards[i] += GAMMA * discounted_rewards[i + 1]
            
            all_returns.append(t.f(discounted_rewards))

        Rs.append(sum(R))
        
        # Prepare batch for training
        S_tensor = t.f(np.array(all_S))
        A_tensor = t.l(np.array(all_A))
        returns_tensor = torch.cat(all_returns, dim=0).flatten()
        old_log_probs_tensor = torch.cat(all_old_log_probs, dim=0).flatten()

        # Train both networks using PPO
        train(S_tensor, A_tensor, returns_tensor, old_log_probs_tensor,
              OPT, V_OPT, PPO_EPOCHS, CLIP_EPSILON)

        # Show mean episodic reward over last 25 episodes
        if Rs: # Avoid division by zero
            mean_r_25 = np.mean(Rs[-25:])
            last25Rs.append(mean_r_25)
            pbar.set_description("R25(%.2f, mean over %d episodes)" % (mean_r_25, len(Rs[-25:])))
        
    pbar.close()
    print("Training finished!")

    # --- 5. Plot and Save Results ---
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True) # More robust folder creation
    
    plot_path = os.path.join(image_dir, f"ppo-{args.mode}.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(last25Rs)), last25Rs, 'b')
    plt.xlabel('Episode')
    plt.ylabel('Reward (averaged over last 25 episodes)')
    plt.title(f"PPO, mode: {args.mode}")
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Episodic reward plot saved to {plot_path}!")

    # --- 6. Testing Loop ---
    print("\nTesting:")
    testRs = []
    for epi in range(TEST_EPISODES):
        S, A, R = envs.play_episode(env, policy, render=False)

        if "mountain_car" in args.mode:
            R = [s[0] for s in S[:-1]]

        testRs.append(sum(R))
        print("Episode%02d: R = %g" % (epi + 1, sum(R)))

    # Print final evaluation score
    if "mountain_car" in args.mode:
        print("Height achieved: %.2f ± %.2f" % (np.mean(testRs), np.std(testRs)))
    else:
        print("Eval score: %.2f ± %.2f" % (np.mean(testRs), np.std(testRs)))

    env.close()

# Standard Python entry point
if __name__ == "__main__":
    main()