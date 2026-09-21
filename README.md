# Reinforced-Learning-2x2-Rubik-s-Cube-Solver

This project aims to solve a 2x2 Rubik's cube through RL, specifically using tabular Q-learning. 

## Files
 **`cube.py`** — The environment for a 2x2 or 3x3 cube. It was built with 3D coordinates, including functions such as `apply_action`, `scramble`,
  `is_solved`, `state_key` (for tabular methods), and `state_vector` (for
  neural nets).

**`q_learning_2x2.py`** — Tabular Q-learning on the 2x2 cube.

**`server.py`** — a local HTTP server. Loads (or trains) the agent on
  startup.
- **`cube_visualizer_local.html`** — the 3D page. Fetches from the endpoints
  above; contains no training code.
- **`run.py`** — starts the server and opens the page in the browser
  automatically. 


## For higher success rate
There are few ways to increase success rate, but it takes longer time and/or bigger space to train:
**In the `q_learning_2x2.py` file**
Increase max_episodes_per_depth (e.g. to 100,000+) s
Decrease success_threshold slightly for deep levels
Increase the steps, e.g. max_steps = depth + 10