import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        V = initialV.copy()
        iterId = 0
        epsilon = np.inf
        
        while iterId < nIterations and epsilon >= tolerance:
            iterId += 1
            V_prev = V.copy()
            
            # T @ V gives the expected future values for each action and state
            # Shape of T: (nActions, nStates, nStates)
            # Shape of V: (nStates,) -> broadcasted to (nActions, nStates, nStates)
            # Result of T @ V: (nActions, nStates)
            Q_values = self.R + self.discount * (self.T @ V)
            
            # For each state, find the action that gives the maximum value
            V = np.max(Q_values, axis=0)
            
            # Calculate the infinity norm of the difference
            epsilon = np.linalg.norm(V - V_prev, np.inf)

        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # Calculate Q-values for all state-action pairs
        Q_values = self.R + self.discount * (self.T @ V)
        
        # For each state, find the index (action) of the maximum Q-value
        policy = np.argmax(Q_values, axis=0)
        
        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi
        which is equivalent to (I - gamma T^pi) V^pi = R^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # Get the reward for the specific policy
        # R_pi[s] = R[policy[s], s]
        R_pi = np.array([self.R[policy[s], s] for s in range(self.nStates)])
        
        # Get the transition probabilities for the specific policy
        # T_pi[s, s'] = T[policy[s], s, s']
        T_pi = np.array([self.T[policy[s], s, :] for s in range(self.nStates)])
        
        # Form the equation (I - gamma * T^pi) V^pi = R^pi
        # A = (I - gamma * T^pi)
        A = np.eye(self.nStates) - self.discount * T_pi
        # b = R^pi
        b = R_pi
        
        # Solve the system of linear equations A * V = b
        V = np.linalg.solve(A, b)
        
        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        policy = initialPolicy.copy()
        V = np.zeros(self.nStates)
        iterId = 0
        policy_stable = False
        
        while iterId < nIterations and not policy_stable:
            iterId += 1
            
            # 1. Policy Evaluation
            V = self.evaluatePolicy(policy)
            
            # 2. Policy Improvement
            new_policy = self.extractPolicy(V)
            
            # Check for convergence
            if np.array_equal(new_policy, policy):
                policy_stable = True
            
            policy = new_policy

        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        V = initialV.copy()
        iterId = 0
        epsilon = np.inf
        
        # Get R^pi and T^pi once, as the policy is fixed
        R_pi = np.array([self.R[policy[s], s] for s in range(self.nStates)])
        T_pi = np.array([self.T[policy[s], s, :] for s in range(self.nStates)])
        
        while iterId < nIterations and epsilon >= tolerance:
            iterId += 1
            V_prev = V.copy()
            
            # Apply Bellman update for the fixed policy
            V = R_pi + self.discount * (T_pi @ V)
            
            epsilon = np.linalg.norm(V - V_prev, np.inf)

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        policy = initialPolicy.copy()
        V = initialV.copy()
        iterId = 0
        epsilon = np.inf
        
        while iterId < nIterations and epsilon >= tolerance:
            iterId += 1
            V_prev = V.copy()
            
            # 1. Partial Policy Evaluation
            V, _, _ = self.evaluatePolicyPartially(policy, V, nIterations=nEvalIterations)
            
            # 2. Policy Improvement
            policy = self.extractPolicy(V)
            
            epsilon = np.linalg.norm(V - V_prev, np.inf)

        return [policy,V,iterId,epsilon]