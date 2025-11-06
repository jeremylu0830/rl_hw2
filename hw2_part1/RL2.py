import numpy as np
import MDP

class RL2:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward
        
    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(float(self.mdp.R[action, state]))
        p = self.mdp.T[action, state, :].astype(float)
        s = p.sum()
        if s <= 0:
            # 退化處理：停在原地（也可用均勻分佈）
            p = np.zeros_like(p); p[state] = 1.0
        else:
            p /= s
        cum = np.cumsum(p)
        r = float(np.random.rand())
        nextState = int(np.searchsorted(cum, r, side="left"))
        return reward, nextState
    
    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm. 
        MODIFIED: Now returns episode_rewards_history for plotting.
        '''

        Q = initialQ.copy()
        N = np.zeros_like(Q)
        
        # 儲存每個 episode 獎勵的 list
        episode_rewards_history = []
        
        for episode in range(nEpisodes):
            currentState = s0
            cumulative_discounted_reward = 0
            
            for step in range(nSteps):
                
                # --- 1. 選擇動作 (Action Selection) ---
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.mdp.nActions)
                else:
                    q_values_for_state = Q[:, currentState]
                    if temperature > 0:
                        # Boltzmann exploration
                        exp_values = np.exp(np.clip(q_values_for_state / temperature, -500, 500)) # clip for numerical stability
                        if np.sum(exp_values) == 0:
                             probs = np.ones(self.mdp.nActions) / self.mdp.nActions # Uniform if all are -inf
                        else:
                             probs = exp_values / np.sum(exp_values)
                        action = np.random.choice(self.mdp.nActions, p=probs)
                    else:
                        # Epsilon-greedy (exploitation part)
                        action = np.argmax(q_values_for_state)

                # --- 2. 與環境互動 ---
                reward, nextState = self.sampleRewardAndNextState(currentState, action)
                
                # 累加折扣後的獎勵
                cumulative_discounted_reward += reward * (self.mdp.discount ** step)

                # --- 3. 更新 Q-table ---
                N[action, currentState] += 1
                # 使用 1/N 的學習率
                alpha = 1.0 / N[action, currentState] 

                max_next_q = np.max(Q[:, nextState])
                target = reward + self.mdp.discount * max_next_q
                Q[action, currentState] += alpha * (target - Q[action, currentState])

                # --- 4. 前進到下一個狀態 ---
                currentState = nextState

            # Episode 結束, 儲存這次的總獎勵
            episode_rewards_history.append(cumulative_discounted_reward)

        policy = np.argmax(Q, axis=0)

        # 回傳 3 個值！
        return [Q, policy, episode_rewards_history]

    # --- Model-Based RL (已修改，會回傳 history) ---
    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning.
        MODIFIED: Now returns episode_rewards_history for plotting.
        '''

        nStates = self.mdp.nStates
        nActions = self.mdp.nActions

        T_hat = defaultT.astype(float).copy()
        R_hat = initialR.astype(float).copy()

        # 確保 T_hat 是一個合法的機率分佈 (防止 MDP __init__ 報錯)
        row_sums = T_hat.sum(axis=2, keepdims=True)
        zero_rows = (row_sums <= 0)
        if np.any(zero_rows):
            idx = np.where(zero_rows)
            # T_hat[idx[0], idx[1], :] = 0.0
            # # 停在原地
            # T_hat[idx[0], idx[1], idx[1]] = 1.0 
            
            T_hat[idx[0], idx[1], :] = 1.0 / nStates
        
        # 歸一化
        T_hat /= T_hat.sum(axis=2, keepdims=True)

        N_sa  = np.zeros((nActions, nStates), dtype=int)
        N_sas = np.zeros((nActions, nStates, nStates), dtype=int)
        R_sum = np.zeros((nActions, nStates), dtype=float)

        learner_mdp = MDP.MDP(T_hat, R_hat, self.mdp.discount) 
        V = np.zeros(nStates, dtype=float)
        policy = np.zeros(nStates, dtype=int)

        # 儲存每個 episode 獎勵的 list
        episode_rewards_history = []

        for episode in range(nEpisodes):
            state = s0
            cumulative_discounted_reward = 0
            
            for step in range(nSteps):
                
                # a. Choose action using epsilon-greedy
                if np.random.rand() < epsilon:
                    action = np.random.randint(nActions) # Explore
                else:
                    action = policy[state] # Exploit
                
                # b. Interact with the REAL environment
                [reward, nextState] = self.sampleRewardAndNextState(state, action)
                
                # 累加折扣後的獎勵
                cumulative_discounted_reward += reward * (self.mdp.discount ** step)

                # c. Update counts and model estimates
                N_sa[action, state] += 1
                N_sas[action, state, nextState] += 1
                R_sum[action, state] += reward
                
                # Update T_hat for this (s, a)
                T_hat[action, state, :] = N_sas[action, state, :] / N_sa[action, state]
                
                # Update R_hat for this (s, a)
                R_hat[action, state] = R_sum[action, state] / N_sa[action, state]
                
                # d. Re-plan: Solve the learned MDP
                learner_mdp.T = T_hat
                learner_mdp.R = R_hat
                
                [V, _, _] = learner_mdp.valueIteration(initialV=V, nIterations=100, tolerance=0.01)
                policy = learner_mdp.extractPolicy(V)
                
                state = nextState
            
            # Episode 結束, 儲存這次的總獎勵
            episode_rewards_history.append(cumulative_discounted_reward)

        # 回傳 3 個值！
        return [V, policy, episode_rewards_history]    

    def epsilonGreedyBandit(self,nIterations):
        ''' Epsilon greedy (MODIFIED: returns reward history) '''
        
        nActions = self.mdp.nActions
        empiricalMeans = np.zeros(nActions)
        actionCounts = np.zeros(nActions, dtype=int)
        
        # 儲存每次迭代 (t) 獲得的獎勵
        reward_history = [] 
        
        for t in range(nIterations):
            epsilon = 1.0 / (t + 1)
            
            if np.random.rand() < epsilon:
                action = np.random.randint(nActions) # Explore
            else:
                action = np.argmax(empiricalMeans) # Exploit
            
            [reward, _] = self.sampleRewardAndNextState(state=0, action=action)
            
            # 將此次獎勵加入 history
            reward_history.append(reward) 
            
            # 更新估計值
            actionCounts[action] += 1
            n = actionCounts[action]
            empiricalMeans[action] += (reward - empiricalMeans[action]) / n

        # 回傳獎勵的歷史紀錄
        return np.array(reward_history)

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        ''' Thompson sampling (MODIFIED: returns reward history) '''
        
        nActions = self.mdp.nActions
        beta_params = prior.copy()
        
        empiricalMeans = np.zeros(nActions)
        actionCounts = np.zeros(nActions, dtype=int)
        
        # 儲存每次迭代 (t) 獲得的獎勵
        reward_history = [] 

        for t in range(nIterations):
            # 1. Sample from posterior
            alphas = beta_params[:, 0].reshape(-1, 1)
            betas = beta_params[:, 1].reshape(-1, 1)
            samples_k = np.random.beta(alphas, betas, size=(nActions, k))
            sampled_thetas = np.mean(samples_k, axis=1)
            
            # 2. Choose action
            action = np.argmax(sampled_thetas)
            
            # 3. Pull arm
            [reward, _] = self.sampleRewardAndNextState(state=0, action=action)
            
            # 將此次獎勵加入 history
            reward_history.append(reward) 

            # 4. Update posterior
            if reward == 1:
                beta_params[action, 0] += 1 # Success
            else:
                beta_params[action, 1] += 1 # Failure
                
            # 5. Update empirical means (for internal use, not returned)
            actionCounts[action] += 1
            n = actionCounts[action]
            empiricalMeans[action] += (reward - empiricalMeans[action]) / n

        # 回傳獎勵的歷史紀錄
        return np.array(reward_history)

    def UCBbandit(self,nIterations):
        ''' UCB (MODIFIED: returns reward history) '''
        
        nActions = self.mdp.nActions
        empiricalMeans = np.zeros(nActions)
        actionCounts = np.zeros(nActions, dtype=int)
        
        # 儲存每次迭代 (t) 獲得的獎勵
        reward_history = [] 

        # 1. Edge case
        if nIterations < nActions:
            for t in range(nIterations):
                action = t
                [reward, _] = self.sampleRewardAndNextState(state=0, action=action)
                reward_history.append(reward) # 紀錄獎勵
                actionCounts[action] = 1
                empiricalMeans[action] = reward
            return np.array(reward_history) # 回傳 history
        
        # 2. Initial pulls
        for a in range(nActions):
            [reward, _] = self.sampleRewardAndNextState(state=0, action=a)
            reward_history.append(reward) # 紀錄獎勵
            actionCounts[a] = 1
            empiricalMeans[a] = reward
            
        # 3. Main UCB loop
        for t in range(nActions, nIterations):
            
            bonus = np.sqrt(2 * np.log(t) / actionCounts)
            ucb_values = empiricalMeans + bonus
            
            action = np.argmax(ucb_values)
            
            [reward, _] = self.sampleRewardAndNextState(state=0, action=action)
            
            reward_history.append(reward) # 紀錄獎勵
            
            # Update
            n = actionCounts[action]
            empiricalMeans[action] = (empiricalMeans[action] * n + reward) / (n + 1)
            actionCounts[action] += 1

        # 回傳獎勵的歷史紀錄
        return np.array(reward_history)