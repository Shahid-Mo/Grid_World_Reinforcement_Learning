# Grid World Reinforcement Learning

This repository demonstrates reinforcement learning algorithms applied to two types of grid world environments: deterministic and stochastic.

## Environments

- `SimpleGridWorldEnv`: A deterministic grid world where the outcome of each action is certain.
- `StochasticGridWorldEnv`: A stochastic version of the grid world where actions can have uncertain outcomes.

## Algorithms

- Q-learning: Demonstrated for both deterministic and stochastic environments.
- SARSA: Demonstrated for the deterministic environment. Implementation can be easily adapted for the stochastic environment.

## Files

- `simple_grid_world_env.py` & `stochastic_grid_world_env.py`: Environment definitions.
- `q_learning.py`: Q-learning algorithm implementation for both environments.
- `sarsa_learning.py`: SARSA algorithm implementation for the deterministic environment.
- `comparison_plot.py`: Compares Q-learning and SARSA performances in the deterministic environment.

## Running the Simulations

Ensure you have the required dependencies installed: `gym`, `numpy`, `matplotlib`.

To run Q-learning on the deterministic and stochastic environment:

```
python q_learning.py
```

To run SARSA on the deterministic and stochastic environment:

```
python sarsa_learning.py
```
To compare Q-learning and SARSA performances:

```
python comparison_plot.py
```

