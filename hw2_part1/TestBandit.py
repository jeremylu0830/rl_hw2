import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt # 繪圖所需

def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# --- 1. MDP Setup ---
# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]]) # 3 actions, 1 state
R = np.array([[0.3],[0.5],[0.7]]) # R[action, state]
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

# --- 2. Experiment Parameters ---
nTrials = 1000
nIterations = 200
k_thompson = 1

# 建立 2D 陣列來儲存所有試驗的獎勵歷史
# (nTrials, nIterations)
ucb_rewards = np.zeros((nTrials, nIterations))
eps_rewards = np.zeros((nTrials, nIterations))
ts_rewards = np.zeros((nTrials, nIterations))

# --- 3. Run Trials ---
print(f"Running {nTrials} trials, each with {nIterations} iterations...")

for trial in range(nTrials):
    if (trial + 1) % 100 == 0:
        print(f"  Completed trial {trial + 1}/{nTrials}")

    # 1. Epsilon-Greedy (epsilon = 1 / # iterations)
    # 假設 RL2.py 中的函式已修改為回傳 reward_history
    eps_rewards[trial, :] = banditProblem.epsilonGreedyBandit(nIterations)
    
    # 2. UCB
    # 假設 RL2.py 中的函式已修改為回傳 reward_history
    ucb_rewards[trial, :] = banditProblem.UCBbandit(nIterations)
    
    # 3. Thompson Sampling
    # Prior: Beta(1, 1) for all arms
    prior = np.ones([mdp.nActions, 2])
    # 假設 RL2.py 中的函式已修改為回傳 reward_history
    ts_rewards[trial, :] = banditProblem.thompsonSamplingBandit(prior, nIterations, k_thompson)

print("All trials complete. Calculating averages...")

# --- 4. Process Results ---
# 沿著 'trial' 軸 (axis=0) 取平均，得到每次迭代 (t) 的平均獎勵
avg_ucb = np.mean(ucb_rewards, axis=0)
avg_eps = np.mean(eps_rewards, axis=0)
avg_ts = np.mean(ts_rewards, axis=0)

# --- 5. Plot the Graph ---
plt.figure(figsize=(12, 7))
plt.plot(avg_eps, label="Epsilon-Greedy ($\epsilon=1/t$)", color='blue', alpha=0.8)
plt.plot(avg_ucb, label="UCB", color='green', alpha=0.8)
plt.plot(avg_ts, label=f"Thompson Sampling (k={k_thompson}, prior=Beta(1,1))", color='red', alpha=0.8)

plt.xlabel("Iteration #")
plt.ylabel(f"Average Reward (over {nTrials} trials)")
plt.title("Bandit Algorithm Comparison (3 Arms: 0.3, 0.5, 0.7)")

# 繪製最佳 arm 的理論平均獎勵 (0.7)
best_arm_reward = np.max(R)
plt.axhline(y=best_arm_reward, color='k', linestyle='--', label=f'Optimal Arm Reward ({best_arm_reward})')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the figure
plot_filename = "bandit_comparison.png"
plt.savefig(plot_filename)
print(f"Graph saved as: {plot_filename}")

plt.show()