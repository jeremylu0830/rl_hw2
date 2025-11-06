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

# REINFORCE
# Slide 10
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-fall21/slides/cs885-module1.pdf


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
            "LEARNING_RATE": 5e-4
        }
    elif "mountain_car" in mode:
        return {
            "OBS_N": 2,
            "ACT_N": 3,
            "ENV_NAME": "MountainCar-v0",
            "GAMMA": 0.9,         # Discount factor
            "LEARNING_RATE": 1e-3
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

# --- Global Constants ---
SEED = 1
EPOCHS = 800              # Total number of iterations to learn over
EPISODES_PER_EPOCH = 1    # Episodes per epoch
TEST_EPISODES = 10        # Test episodes
HIDDEN = 32               # Hidden size
POLICY_TRAIN_ITERS = 1    # Number of iterations of policy improvement in each epoch

# Setup device
t = utils.torch.TorchHelper()
DEVICE = t.device

# --- Global Network Definition ---
# (Must be global for the `policy` function to access it)
# We initialize it with dummy values; it will be re-initialized in main()
# This is a common pattern to allow top-level function definitions
pi = torch.nn.Sequential(torch.nn.Linear(1, 1)).to(DEVICE)

def policy(env, obs: np.ndarray | tuple | dict) -> int:
    """
    Selects an action based on the policy network given an observation.
    Handles tuple observations from gymnasium.
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


def train(S: torch.Tensor, A: torch.Tensor, returns: torch.Tensor, GAMMA: float, OPT: torch.optim.Optimizer):
    """
    Performs one step of policy gradient update using the REINFORCE objective.
    (This function uses the global `pi` network)
    """
    for _ in range(POLICY_TRAIN_ITERS):
        OPT.zero_grad()

        # Get log probabilities: log(pi(a_t | s_t))
        log_probs = torch.nn.LogSoftmax(dim=-1)(pi(S)).gather(1, A.view(-1, 1)).view(-1)
        
        # Create a tensor of timesteps [0, 1, 2, ...]
        n = torch.arange(S.size(0)).to(DEVICE)
        
        # Calculate the REINFORCE objective
        # Note: This implementation applies an *additional* discount (GAMMA**n)
        objective = -((GAMMA**n) * returns * log_probs).sum()
        
        objective.backward()
        OPT.step()


def main():
    """Main execution block for training and evaluation."""
    global pi # Declare intent to modify the global policy network
    
    args = setup_parser()
    
    # --- 1. Setup Configuration ---
    config = get_env_config(args.mode)
    
    OBS_N = config["OBS_N"]
    ACT_N = config["ACT_N"]
    ENV_NAME = config["ENV_NAME"]
    GAMMA = config["GAMMA"]
    LEARNING_RATE = config["LEARNING_RATE"]
    
    # --- 2. Initialize Environment and Seed ---
    utils.seed.seed(SEED)
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    # Seed the environment during the first reset
    env.reset(seed=SEED) 
    
    # --- 3. Initialize Network and Optimizer ---
    # (Re-initialize the global `pi` with correct dimensions)
    pi = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    
    OPT = torch.optim.Adam(pi.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    Rs = []
    last25Rs = []
    print("Training:")
    pbar = tqdm.trange(EPOCHS)
    for epi in pbar:

        all_S, all_A = [], []
        all_returns = []

        for _ in range(EPISODES_PER_EPOCH):
            
            # Play an episode
            S, A, R = envs.play_episode(env, policy)

            # Modify reward for "mountain_car_mod"
            if args.mode == "mountain_car_mod":
                R = [s[0] for s in S[:-1]]

            all_S += S[:-1]  # ignore last state
            all_A += A
            
            # Create returns (G_t = "return-to-go")
            discounted_rewards = copy.deepcopy(R)
            for i in range(len(R) - 1)[::-1]:
                discounted_rewards[i] += GAMMA * discounted_rewards[i + 1]
            
            all_returns.append(t.f(discounted_rewards))

        Rs.append(sum(R))
        
        # Prepare data for training
        S_tensor = t.f(np.array(all_S))
        A_tensor = t.l(np.array(all_A))
        returns_tensor = torch.cat(all_returns, dim=0).flatten()

        # Train the policy network
        train(S_tensor, A_tensor, returns_tensor, GAMMA, OPT)

        # Show mean episodic reward over last 25 episodes
        if Rs: # Avoid division by zero
            last25Rs.append(np.mean(Rs[-25:]))
            pbar.set_description("R25(%.2f, mean over %d episodes)" % (last25Rs[-1], len(Rs[-25:])))
        
    pbar.close()
    print("Training finished!")

    # --- 5. Plot and Save Results ---
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True) # More robust folder creation
    
    plot_path = os.path.join(image_dir, f"reinforce-{args.mode}.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(last25Rs)), last25Rs, 'b')
    plt.xlabel('Episode')
    plt.ylabel('Reward (averaged over last 25 episodes)')
    plt.title(f"REINFORCE, mode: {args.mode}")
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