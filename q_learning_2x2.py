
import random
import json
from collections import defaultdict

from cube import Cube

N_ACTIONS = len(Cube.ACTIONS)  # 12: 6 faces x {clockwise, counter-clockwise}

def action_to_string(action):
    face, clockwise = Cube.ACTIONS[action]

    if clockwise:
        return face
    else:
        return face + "'"

class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.3):
        self.q = defaultdict(float)   # (state_key, action_idx) -> value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _qvals(self, state_key):
        return [self.q[(state_key, a)] for a in range(N_ACTIONS)]

    def choose_action(self, state_key, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        qvals = self._qvals(state_key)
        best = max(range(N_ACTIONS), key=lambda a: qvals[a])
        return best

    def update(self, state_key, action, reward, next_key, done):
        target = reward
        if not done:
            target += self.gamma * max(self._qvals(next_key))
        td_error = target - self.q[(state_key, action)]
        self.q[(state_key, action)] += self.alpha * td_error


SOLVE_REWARD = 20.0
STEP_REWARD = -1.0


def run_episode(agent: QLearningAgent, depth: int, max_steps: int, train: bool):
    cube = Cube(2)
    cube.scramble(depth)
    state_key = cube.state_key()

    if cube.is_solved():  # depth-0 edge case
        return True, 0

    for step in range(max_steps):
        action = agent.choose_action(state_key, greedy=not train)
        cube.apply_action(action)
        next_key = cube.state_key()
        solved = cube.is_solved()
        reward = SOLVE_REWARD if solved else STEP_REWARD

        if train:
            agent.update(state_key, action, reward, next_key, solved)

        state_key = next_key
        if solved:
            return True, step + 1

    return False, max_steps


def evaluate(agent, depth, n_episodes=200):
    successes, total_steps = 0, 0
    for _ in range(n_episodes):
        solved, steps = run_episode(agent, depth, max_steps=depth + 6, train=False)
        successes += solved
        total_steps += steps
    return successes / n_episodes, total_steps / n_episodes

def solve_and_save(agent, depth=8, filename="solution.json"):
    """for visualizer: scramble a cube, solve it with the agent, and save the moves to a JSON file"""

    cube = Cube(2)

    # Generate the scramble ourselves so we know exactly what moves were used.
    scramble = []
    for _ in range(depth):
        action = random.randrange(N_ACTIONS)
        scramble.append(action)
        cube.apply_action(action)

    solution = []
    state_key = cube.state_key()

    max_steps = depth + 10

    for step in range(max_steps):
        if cube.is_solved():
            break

        # greedy=True means NO random exploration
        action = agent.choose_action(state_key, greedy=True)

        cube.apply_action(action)

        solution.append(action)

        state_key = cube.state_key()

    data = {
    "scramble": scramble,
    "solution": solution,

    "scramble_moves": [
        action_to_string(a) for a in scramble
    ],

    "solution_moves": [
        action_to_string(a) for a in solution
    ]
}

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print("\nVisualization data saved!")
    print("Scramble:", " ".join(data["scramble"]))
    print("Solution:", " ".join(data["solution"]))
    print("Solved:", cube.is_solved())

    return data

def train_curriculum(
    max_depth=8,
    episodes_per_check=500,
    success_threshold=0.9,
    max_episodes_per_depth=20000,
):
    agent = QLearningAgent()
    depth = 1
    total_episodes = 0

    while depth <= max_depth:
        episodes_this_depth = 0
        while episodes_this_depth < max_episodes_per_depth:
            for _ in range(episodes_per_check):
                run_episode(agent, depth, max_steps=depth + 6, train=True)
            episodes_this_depth += episodes_per_check
            total_episodes += episodes_per_check

            success_rate, avg_steps = evaluate(agent, depth)
            print(
                f"depth={depth:2d}  episodes_at_depth={episodes_this_depth:6d}  "
                f"total_episodes={total_episodes:7d}  "
                f"success_rate={success_rate:5.1%}  avg_steps={avg_steps:4.1f}  "
                f"Q_table_size={len(agent.q):7d}"
            )
            if success_rate >= success_threshold:
                break

        depth += 1

    return agent


if __name__ == "__main__":
    print("Training agent on the 2x2 cube with curriculum "
          "scrambling...\n")

    agent = train_curriculum(max_depth=8)

    print("\nFinal evaluation across depths:")
    for d in range(1, 9):
        rate, steps = evaluate(agent, d, n_episodes=300)
        print(
            f"  depth={d}: success_rate={rate:5.1%} "
            f"avg_steps={steps:4.1f}"
        )

    solve_and_save(agent, depth=8)