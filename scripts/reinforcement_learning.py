import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple environment
class DNAEnvironment(gym.Env):
    def __init__(self):
        super(DNAEnvironment, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(100,))
        self.action_space = gym.spaces.Discrete(2)
        self.state = np.random.rand(100)

    def reset(self):
        self.state = np.random.rand(100)
        return self.state

    def step(self, action):
        reward = -np.sum(np.abs(self.state - action))  # Dummy reward
        self.state = np.random.rand(100)
        done = False
        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

# Define a simple neural network for the agent
class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function for the agent
def train_agent(episodes=1000, learning_rate=0.001):
    env = DNAEnvironment()
    agent = DQNAgent()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(100):  # Limit steps per episode
            state_tensor = torch.FloatTensor(state)
            q_values = agent(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            next_state_tensor = torch.FloatTensor(next_state)
            target = reward + 0.99 * torch.max(agent(next_state_tensor))
            target_f = agent(state_tensor)
            target_f[action] = target

            optimizer.zero_grad()
            loss = criterion(agent(state_tensor), target_f)
            loss.backward()
            optimizer.step()

            if done:
                break

        if episode % 100 == 0:
            print(f'Episode [{episode}/{episodes}], Total Reward: {total_reward}')

    torch.save(agent.state_dict(), 'models/trained_model.pth')

# Example usage
if __name__ == '__main__':
    train_agent()
