import numpy as np
import matplotlib.pyplot as plt

# Multi-arm Bandits Class
class MAB:
    # initialization
    def __init__(self, K, thetas=None):
        if thetas == None:
            thetas = np.random.uniform(0, 1, size=(K, 1))
        assert K == thetas.shape[0], "Thetas dim do not match K."
        # store prob of each bandit
        self.thetas = thetas

    # function that helps us draw from the bandits
    def draw(self, k):
        # we return the reward and the regret of the action
        reward = np.random.binomial(1, self.thetas[k])
        regret = np.max(self.thetas) - self.thetas[k]
        return reward, regret

# Exploitation/Exploration Policies
class eGreedy:
    # select the best greedy action with probablity epsilon
    def __init__(self, epsilon, K, draws=None, rewards=None, MAB=None):
        self.epsilon = epsilon
        # assumes beta(1,1) prior
        if draws == None:
            draws = np.ones((K,)) * 2
        if rewards == None:
            rewards = np.ones((K,))
        self.K = K
        self.draws = draws
        self.rewards = rewards
        if MAB == None:
            self.MAB = MAB(K)
        else:
            self.MAB = MAB
        self.regret = []
    def choose(self):
        # reward-pull ratio for each bandit
        ratios = self.rewards / self.draws
        # explore vs exploit
        if np.random.random() < self.epsilon:
            bandits = np.delete(np.arange(self.K), np.argmax(ratios))
            c = np.random.choice(bandits)
        else:
            c = np.argmax(ratios)
        reward, regret = self.MAB.draw(c)
        self.regret.append(regret)
        self.rewards[c] += reward
        self.draws[c] += 1
    def est_theta(self):
        return self.rewards / self.draws

class UpperConfidenceBound:
    def __init__(self, K, draws=None, rewards=None, MAB=None):
        # assumes beta(1,1) prior
        if draws == None:
            draws = np.ones((K,)) * 2
        if rewards == None:
            rewards = np.ones((K,))
        self.K = K
        self.draws = draws
        self.rewards = rewards
        if MAB == None:
            self.MAB = MAB(K)
        else:
            self.MAB = MAB
        self.regret = []
    def choose(self):
        # reward-pull ratio for each bandit
        ratios = self.rewards / self.draws
        # explore vs exploit
        sqrt_term = np.sqrt(2 * np.log(np.sum(self.draws)) / self.draws)
        c = np.argmax(ratios + sqrt_term)
        reward, regret = self.MAB.draw(c)
        self.regret.append(regret)
        self.rewards[c] += reward
        self.draws[c] += 1
    def est_theta(self):
        return self.rewards / self.draws

class Thompson:
    # At each round, we want to pick a bandit with probability equal to the probability of it being the optimal choice.
    def __init__(self, K, draws=None, rewards=None, MAB=None):
        # assumes beta(1,1) prior
        if draws == None:
            draws = np.ones((K,)) * 2
        if rewards == None:
            rewards = np.ones((K,))
        self.K = K
        self.draws = draws
        self.rewards = rewards
        if MAB == None:
            self.MAB = MAB(K)
        else:
            self.MAB = MAB
        self.regret = []
    def choose(self):
        failures = self.draws - self.rewards
        samples = np.array([np.random.beta(1 + self.rewards[id], 1 + failures[id]) for id in range(self.K)])
        c = np.argmax(samples)
        reward, regret = self.MAB.draw(c)
        self.regret.append(regret[0])
        self.rewards[c] += reward
        self.draws[c] += 1
    def est_theta(self):
        return self.rewards / self.draws

# Simulations
def RegretSim(pulls=800, K=40):
    M = MAB(K)
    t = Thompson(K, MAB=M)
    ucb = UpperConfidenceBound(K, MAB=M)
    eG = eGreedy(0.1, K, MAB=M)
    for _ in range(pulls):
        t.choose()
        ucb.choose()
        eG.choose()
    plt.plot(np.arange(pulls), np.cumsum(t.regret), label='Thompson')
    plt.plot(np.arange(pulls), np.cumsum(ucb.regret), label='UCB')
    plt.plot(np.arange(pulls), np.cumsum(eG.regret), label='Greedy')
    plt.legend()
    plt.title('Accumulated Regret for Three MAB Policies')
    plt.ylabel('Regret')
    plt.xlabel('Trials')
    plt.show()

if __name__ == '__main__':
    RegretSim()