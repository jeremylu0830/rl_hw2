# 強化學習作業二：Multi-Bandits、Model-Based vs. Q-Learning、Policy Gradients 與 PPO

> **課程：** [CS885 強化學習](https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/assignments.html)

---

## 🧪 實驗一｜Model-Based RL vs. Q-Learning（迷宮）

<img width="3570" height="2066" alt="maze" src="https://github.com/user-attachments/assets/ad9d1863-aaeb-41ef-9f30-d56ea7a1c0e2" />

### 📈 實驗觀察

* **Model-Based RL（紅線）**：學習曲線上升極快，約在 **Episode ≈ 25** 即接近最佳平均獎勵（≈ 50），收斂速度極佳。
* **Q-Learning（藍線）**：學習曲線較平緩，約在 **Episode ≈ 75** 才追上 Model-Based RL 的獎勵水準。
* **共同點**：75 個 episodes 後兩者最終性能相近；兩條曲線皆有明顯尖銳震盪，源於 **ε = 0.05** 的隨機探索。

### 🧠 演算法機制與樣本效率

#### A. Model-Based RL（高樣本效率）

* **思想**：同時學得環境模型 **( \hat T, \hat R )**，並在更新後立即 **規劃（Planning）**（例如 Value Iteration）。
* **資訊傳播**：一旦隨機探索到終點的 **+100** 獎勵，規劃即可將價值訊息「反向擴散」至大量狀態，**不必反覆到達終點** 也能快速改善策略。
* **結論**：(V) 與 policy 迅速收斂，樣本效率高。

#### B. Q-Learning（較低樣本效率）

* **思想**：無模型（Model-Free），僅以經驗更新 **Q-Table**（一次只更新 **Q[a, s]** 一格）。
* **資訊傳播**：獎勵僅能逐格向前擴散；終點前兩格、三格…需多次造訪才能間接學到，**收斂較慢**。
* **結論**：價值資訊像漣漪般逐步回傳，學習曲線上升緩慢。

---

## 🎰 實驗二｜Bandit：Epsilon-Greedy、UCB、Thompson Sampling
<img width="1200" height="700" alt="bandit_comparison" src="https://github.com/user-attachments/assets/a20ae87e-abad-42e3-9301-29dcadf3ed65" />

### 📈 實驗觀察

以 1000 次重複試驗、每次 200 迭代的平均獎勵為指標；虛線（0.7）為最佳手臂真值。

* **Thompson Sampling（紅）**：表現最佳，**上升最快且最穩定**，最接近 0.7。
* **UCB（綠）**：次佳，明顯優於 Epsilon-Greedy。
* **Epsilon-Greedy（藍）**：最慢、最差，至結束仍與最佳獎勵有距離。

### 2. 為何結果符合理論

* **Epsilon-Greedy（ε = 1/t）**：**無方向**的隨機探索；即便幾乎能確定某臂很差，仍以機率拉動，浪費樣本。
* **UCB**：**樂觀探索**，以 ( \sqrt{2,\log t / N_a} ) 作為置信上界加成，主動彌補不確定性（(N_a) 小時被鼓勵探索）。
* **Thompson Sampling**：**貝氏抽樣**，對每臂維持信念分布（如 Beta），抽樣的 (\theta) 同時兼顧「好臂利用」與「不確定臂探索」，自然平衡 **Explore/Exploit**，因此收斂最快。

---

## 🎯 實驗三｜REINFORCE、REINFORCE w/ Baseline、PPO（CartPole-v1）
<img width="1000" height="600" alt="reinforce-cartpole" src="https://github.com/user-attachments/assets/981ed1c6-1101-4c85-bab9-fe582aa4940c" />
<img width="1000" height="600" alt="reinforce-baseline-cartpole" src="https://github.com/user-attachments/assets/e3bed63c-fe83-4dc0-81b4-ee4c34936039" />
<img width="1000" height="600" alt="ppo-cartpole" src="https://github.com/user-attachments/assets/4b3af693-12d1-4005-9431-63dec08a4db5" />

### 📈 實驗觀察

* **REINFORCE**（0–800 episodes）：可學但**高度不穩定**；約 400 開始起飛，550–600 達到最大奖勵（≈ 200）。
* **REINFORCE w/ Baseline**（0–800）：曲線更**平滑**；約 400 起飛，最終收斂到最大奖勵。
* **PPO**（0–150）：**極高效率且穩定**；40–50 迅速上升，120–130 即達最大奖勵。

### 📉 變異性與穩定性

* **REINFORCE**：以全回報 (G_t) 作估計，變異性高 → 梯度噪聲大 → 震盪明顯。
* **Baseline**：引入 **Advantage**（(A = G_t - V(s))）降低方差 → 梯度更穩，曲線更平滑。
* **PPO**：

  * **裁剪目標**：限制新舊策略差距，避免崩潰。
  * **多 Epoch 更新**：同批資料反覆利用，提高樣本效率。

### 🔁 關於 **POLICY_TRAIN_ITERS**

* **REINFORCE 類（On-Policy）**：資料必須來自**當前策略**。將 **POLICY_TRAIN_ITERS** 由 1 提高到 10，等同於用「過時資料」做多次更新 → **梯度估計偏差、更新過大** → 容易崩潰。
* **PPO**：裁剪機制允許多次重用同批資料而不致發散，因而樣本效率高。

---

## 🏔️ 實驗四｜Mountain-Car：為何在原版環境失敗？
<img width="1000" height="600" alt="reinforce-mountain_car" src="https://github.com/user-attachments/assets/8f2173c3-2992-4ec9-9c98-ac91735d5c37" />
<img width="1000" height="600" alt="reinforce-baseline-mountain_car" src="https://github.com/user-attachments/assets/c93626e3-8ea6-4c33-80a5-11fdc0fb1b78" />
<img width="1000" height="600" alt="ppo-mountain_car" src="https://github.com/user-attachments/assets/112038f5-f9fb-4a1e-b6df-e4eecb27ac11" />

### ⚖️ CartPole vs. Mountain-Car

* **CartPole**：所有方法最終成功（PPO ≫ Baseline ≫ REINFORCE）。
* **Mountain-Car（原版）**：三者均失敗，總回報 **-200** 橫線（200 步超時、每步 -1）。

### 🧨 失敗主因：**獎勵稀疏 → 零梯度**

* **稀疏獎勵**：未到旗幟前皆為 -1，且隨機策略幾乎不可能「偶然」完成先退後衝的複雜策略。
* **回報無變異**：所有軌跡回報皆約 -200 → (G_t) 與 (A) 幾乎為常數 → **梯度 ≈ 0**，無法學習。

---

## 🏔️✨ 實驗五｜Modified Mountain-Car（高度獎勵）
<img width="1000" height="600" alt="reinforce-mountain_car_mod" src="https://github.com/user-attachments/assets/b8c91e53-aad4-433e-94ee-26562af77d92" />
<img width="1000" height="600" alt="reinforce-baseline-mountain_car" src="https://github.com/user-attachments/assets/5e6f9958-adfa-4558-b101-500d1ed28434" />
<img width="1000" height="600" alt="ppo-mountain_car_mod" src="https://github.com/user-attachments/assets/70d8c8dc-78e6-4877-9f9b-8588b2b66b0d" />

### 📈 實驗觀察

* **REINFORCE**：開始學習但仍高變異；整體緩慢上升（約 -103 → -99）。
* **REINFORCE w/ Baseline**：明顯**穩定進步**（約 -100 → -82.5）。
* **PPO**：**快速且平滑**，約 70 episodes 即近最優（≈ -75）。

### 🔎 根本原因：**獎勵塑形（Reward Shaping）**

* **原版**：稀疏獎勵 → 零梯度 → 全面失敗。
* **改良版（高度作為稠密獎勵）**：每步都有意義回饋 → 回報具變異 → 產生**清晰梯度** → 演算法順利學習。

---

