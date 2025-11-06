import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt

''' 
Construct a simple maze MDP (迷宮 MDP)
(原有的 MDP T, R, discount 設定保持不變)
'''

# Transition function: |A| x |S| x |S'| array
T = np.zeros([4,17,17])
a = 0.8;  # intended move
b = 0.1;  # lateral move

# up (a = 0)
T[0,0,0] = a+b; T[0,0,1] = b;
T[0,1,0] = b; T[0,1,1] = a; T[0,1,2] = b;
T[0,2,1] = b; T[0,2,2] = a; T[0,2,3] = b;
T[0,3,2] = b; T[0,3,3] = a+b;
T[0,4,4] = b; T[0,4,0] = a; T[0,4,5] = b;
T[0,5,4] = b; T[0,5,1] = a; T[0,5,6] = b;
T[0,6,5] = b; T[0,6,2] = a; T[0,6,7] = b;
T[0,7,6] = b; T[0,7,3] = a; T[0,7,7] = b;
T[0,8,8] = b; T[0,8,4] = a; T[0,8,9] = b;
T[0,9,8] = b; T[0,9,5] = a; T[0,9,10] = b;
T[0,10,9] = b; T[0,10,6] = a; T[0,10,11] = b;
T[0,11,10] = b; T[0,11,7] = a; T[0,11,11] = b;
T[0,12,12] = b; T[0,12,8] = a; T[0,12,13] = b;
T[0,13,12] = b; T[0,13,9] = a; T[0,13,14] = b;
T[0,14,16] = 1;
T[0,15,11] = a; T[0,15,14] = b; T[0,15,15] = b;
T[0,16,16] = 1;

# down (a = 1)
T[1,0,0] = b; T[1,0,4] = a; T[1,0,1] = b;
T[1,1,0] = b; T[1,1,5] = a; T[1,1,2] = b;
T[1,2,1] = b; T[1,2,6] = a; T[1,2,3] = b;
T[1,3,2] = b; T[1,3,7] = a; T[1,3,3] = b;
T[1,4,4] = b; T[1,4,8] = a; T[1,4,5] = b;
T[1,5,4] = b; T[1,5,9] = a; T[1,5,6] = b;
T[1,6,5] = b; T[1,6,10] = a; T[1,6,7] = b;
T[1,7,6] = b; T[1,7,11] = a; T[1,7,7] = b;
T[1,8,8] = b; T[1,8,12] = a; T[1,8,9] = b;
T[1,9,8] = b; T[1,9,13] = a; T[1,9,10] = b;
T[1,10,9] = b; T[1,10,14] = a; T[1,10,11] = b;
T[1,11,10] = b; T[1,11,15] = a; T[1,11,11] = b;
T[1,12,12] = a+b; T[1,12,13] = b;
T[1,13,12] = b; T[1,13,13] = a; T[1,13,14] = b;
T[1,14,16] = 1;
T[1,15,14] = b; T[1,15,15] = a+b;
T[1,16,16] = 1;

# left (a = 2)
T[2,0,0] = a+b; T[2,0,4] = b;
T[2,1,1] = b; T[2,1,0] = a; T[2,1,5] = b;
T[2,2,2] = b; T[2,2,1] = a; T[2,2,6] = b;
T[2,3,3] = b; T[2,3,2] = a; T[2,3,7] = b;
T[2,4,0] = b; T[2,4,4] = a; T[2,4,8] = b;
T[2,5,1] = b; T[2,5,4] = a; T[2,5,9] = b;
T[2,6,2] = b; T[2,6,5] = a; T[2,6,10] = b;
T[2,7,3] = b; T[2,7,6] = a; T[2,7,11] = b;
T[2,8,4] = b; T[2,8,8] = a; T[2,8,12] = b;
T[2,9,5] = b; T[2,9,8] = a; T[2,9,13] = b;
T[2,10,6] = b; T[2,10,9] = a; T[2,10,14] = b;
T[2,11,7] = b; T[2,11,10] = a; T[2,11,15] = b;
T[2,12,8] = b; T[2,12,12] = a+b;
T[2,13,9] = b; T[2,13,12] = a; T[2,13,13] = b;
T[2,14,16] = 1;
T[2,15,11] = a; T[2,15,14] = b; T[2,15,15] = b;
T[2,16,16] = 1;

# right (a = 3)
T[3,0,0] = b; T[3,0,1] = a; T[3,0,4] = b;
T[3,1,1] = b; T[3,1,2] = a; T[3,1,5] = b;
T[3,2,2] = b; T[3,2,3] = a; T[3,2,6] = b;
T[3,3,3] = a+b; T[3,3,7] = b;
T[3,4,0] = b; T[3,4,5] = a; T[3,4,8] = b;
T[3,5,1] = b; T[3,5,6] = a; T[3,5,9] = b;
T[3,6,2] = b; T[3,6,7] = a; T[3,6,10] = b;
T[3,7,3] = b; T[3,7,7] = a; T[3,7,11] = b;
T[3,8,4] = b; T[3,8,9] = a; T[3,8,12] = b;
T[3,9,5] = b; T[3,9,10] = a; T[3,9,13] = b;
T[3,10,6] = b; T[3,10,11] = a; T[3,10,14] = b;
T[3,11,7] = b; T[3,11,11] = a; T[3,11,15] = b;
T[3,12,8] = b; T[3,12,13] = a; T[3,12,12] = b;
T[3,13,9] = b; T[3,13,14] = a; T[3,13,13] = b;
T[3,14,16] = 1;
T[3,15,11] = b; T[3,15,15] = a+b;
T[3,16,16] = 1;

# Reward function: |A| x |S| array
R = -1 * np.ones([4,17]);
R[:,14] = 100;  # goal state
R[:,9] = -70;   # bad state
R[:,16] = 0;    # end state

# Discount factor: scalar in [0,1)
discount = 0.95
        
# MDP object
mdp = MDP.MDP(T,R,discount)

# RL problem
# 使用 np.random.normal 來抽樣獎勵 (雖然在這個確定性獎勵的迷宮中, 影響不大)
rlProblem = RL2.RL2(mdp, lambda mu: mu) 
# 如果希望獎勵有隨機性, 可以用:
# rlProblem = RL2.RL2(mdp, np.random.normal) 

# --- 實驗參數設定 ---
nTrials = 100
nEpisodes = 200
nSteps = 100
epsilon = 0.05
s0 = 0 # 起始狀態

print(f"開始執行 {nTrials} 次試驗...")
print(f"每_次試驗: {nEpisodes} episodes, 每_episode: {nSteps} steps.")

# --- 初始化儲存獎勵紀錄的陣列 ---
# (nTrials, nEpisodes)
q_learning_histories = np.zeros((nTrials, nEpisodes))
model_based_histories = np.zeros((nTrials, nEpisodes))

# --- 主迴圈 (執行 nTrials 次試驗) ---
for trial in range(nTrials):
    if (trial + 1) % 10 == 0:
        print(f"  正在執行試驗 {trial + 1}/{nTrials}...")

    # --- 1. 執行 Model-Based RL ---
    # 根據題目要求
    defaultT = np.ones([mdp.nActions, mdp.nStates, mdp.nStates]) / mdp.nStates
    initialR = np.zeros([mdp.nActions, mdp.nStates])
    
    [V_mb, policy_mb, mb_history] = rlProblem.modelBasedRL(
        s0=s0,
        defaultT=defaultT,
        initialR=initialR,
        nEpisodes=nEpisodes,
        nSteps=nSteps,
        epsilon=epsilon
    )
    # 儲存此次試驗的獎勵紀錄
    model_based_histories[trial, :] = mb_history

    # --- 2. 執行 Q-Learning ---
    # 根據題目要求
    initialQ = np.zeros([mdp.nActions, mdp.nStates])
    
    [Q_q, policy_q, q_history] = rlProblem.qLearning(
        s0=s0,
        initialQ=initialQ,
        nEpisodes=nEpisodes,
        nSteps=nSteps,
        epsilon=epsilon,
        temperature=0 # 確保使用 epsilon-greedy
    )
    # 儲存此次試驗的獎勵紀錄
    q_learning_histories[trial, :] = q_history

print("所有試驗執行完畢。")

# --- 3. 計算 100 次試驗的平均 ---
# 沿著 'trial' 軸 (axis=0) 取平均
avg_q_rewards = np.mean(q_learning_histories, axis=0)
avg_mb_rewards = np.mean(model_based_histories, axis=0)

# --- 4. 繪製結果圖表 ---
print("正在繪製圖表...")
plt.figure(figsize=(12, 7))

# 繪製 Q-Learning 曲線
plt.plot(avg_q_rewards, label=f"Q-Learning (epsilon={epsilon})", color='blue')

# 繪製 Model-Based RL 曲線
plt.plot(avg_mb_rewards, label=f"Model-Based RL (epsilon={epsilon})", color='red')

plt.xlabel("Episode")
plt.ylabel("Average Cumulative Discounted Reward")
plt.title(f"Model-Based RL vs. Q-Learning (Average over {nTrials} Trials)")
plt.legend()
plt.grid(True)
plt.show()

# Save the figure
plot_filename = "maze.png"
plt.savefig(plot_filename)
print(f"Graph saved as: {plot_filename}")