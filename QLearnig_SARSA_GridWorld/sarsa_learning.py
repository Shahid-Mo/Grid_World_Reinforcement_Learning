import numpy as np
import pickle
from QLearnig_SARSA_GridWorld.simple_grid_world_env import SimpleGridWorldEnv
from stochastic_grid_world_env import StochasticGridWorldEnv 


def sarsa_learning(env, num_episodes=500, alpha=0.1, gamma=1.0, initial_epsilon=1.0, min_epsilon=0.01):
    epsilon_decay = (min_epsilon / initial_epsilon) ** (1 / num_episodes)
    q_table = np.zeros((env.num_states, env.num_actions))
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        eps = max(min_epsilon, initial_epsilon * (epsilon_decay ** episode))
        action = np.random.choice(env.num_actions) if np.random.rand() < eps else np.argmax(q_table[state])

        while not done:
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            next_action = np.random.choice(env.num_actions) if np.random.rand() < eps else np.argmax(q_table[next_state])

            # SARSA update
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

            state, action = next_state, next_action

        total_rewards.append(total_reward)

    return q_table, total_rewards

if __name__ == "__main__":
    # Example usage with both environments
    deterministic_env = SimpleGridWorldEnv()
    stochastic_env = StochasticGridWorldEnv()
    
    print("Training with deterministic environment...")
    q_table_deterministic = sarsa_learning(deterministic_env)
    with open("deterministic_sarsa.pkl", "wb") as f:
        pickle.dump(q_table_deterministic, f)
    
    print("\nTraining with stochastic environment...")
    q_table_stochastic = sarsa_learning(stochastic_env)
    with open("stochastic_sarsa.pkl", "wb") as f:
        pickle.dump(q_table_stochastic, f)