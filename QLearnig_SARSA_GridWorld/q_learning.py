import numpy as np
import pickle
from QLearnig_SARSA_GridWorld.simple_grid_world_env import SimpleGridWorldEnv
from stochastic_grid_world_env import StochasticGridWorldEnv 

def q_learning(env, num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.6, min_epsilon=0.01):
    epsilon_decay = (min_epsilon / epsilon) ** (1 / num_episodes)
    q_table = np.zeros((env.num_states, env.num_actions))
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        eps = max(min_epsilon, epsilon * (epsilon_decay ** episode))

        while not done:
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Q-Learning update
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            state = next_state

        total_rewards.append(total_reward)

    return q_table, total_rewards

if __name__ == "__main__":
    # Example usage with both environments
    deterministic_env = SimpleGridWorldEnv()
    stochastic_env = StochasticGridWorldEnv()
    
    print("Training with deterministic environment...")
    q_table_deterministic = q_learning(deterministic_env)
    with open("deterministic_q_table.pkl", "wb") as f:
        pickle.dump(q_table_deterministic, f)
    
    print("\nTraining with stochastic environment...")
    q_table_stochastic = q_learning(stochastic_env)
    with open("stochastic_q_table.pkl", "wb") as f:
        pickle.dump(q_table_stochastic, f)