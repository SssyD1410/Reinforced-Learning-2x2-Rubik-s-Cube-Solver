import json
import os
import random
import time
from collections import defaultdict

from cube import Cube

FACES = Cube.FACES
ACTIONS = Cube.ACTIONS       # same face/direction order as the JS ACTIONS list
N_ACTIONS = len(ACTIONS)

SOLVE_REWARD = 20.0
STEP_REWARD = -1.0


# Fixed, canonical order of the 8 corner position slots. Both this script
# and the JS in cube_visualizer_local.html iterate slots in this exact
# order, so a state can be encoded as just "3 color letters per slot" --
# no coordinates or direction vectors need to be stored in the key at all.
POSITION_ORDER = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]


def js_state_key(cube: Cube) -> str:
    """
    Compact canonical state string (24 characters): for each of the 8
    position slots, in POSITION_ORDER, the 3 sticker colors currently
    showing on that slot's x/y/z faces. 
    """
    pos_map = {c.pos: c for c in cube.cubies}
    parts = []
    for slot in POSITION_ORDER:
        cubie = pos_map[slot]
        cx = cubie.stickers[(slot[0], 0, 0)]
        cy = cubie.stickers[(0, slot[1], 0)]
        cz = cubie.stickers[(0, 0, slot[2])]
        parts.append(cx + cy + cz)
    return ''.join(parts)


class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.3):
        self.q = defaultdict(float)     # (state_key, action_idx) -> value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _qvals(self, key):
        return [self.q[(key, a)] for a in range(N_ACTIONS)]

    def choose_action(self, key, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        qvals = self._qvals(key)
        return max(range(N_ACTIONS), key=lambda a: qvals[a])

    def update(self, key, action, reward, next_key, done):
        target = reward
        if not done:
            target += self.gamma * max(self._qvals(next_key))
        self.q[(key, action)] += self.alpha * (target - self.q[(key, action)])


def run_episode(agent, depth, max_steps, train):
    cube = Cube(2)
    cube.scramble(depth)
    key = js_state_key(cube)
    if cube.is_solved():
        return True, 0
    for step in range(max_steps):
        action = agent.choose_action(key, greedy=not train)
        face, cw = ACTIONS[action]
        cube.apply_move(face, cw)
        next_key = js_state_key(cube)
        solved = cube.is_solved()
        reward = SOLVE_REWARD if solved else STEP_REWARD
        if train:
            agent.update(key, action, reward, next_key, solved)
        key = next_key
        if solved:
            return True, step + 1
    return False, max_steps


def evaluate(agent, depth, n=150):
    succ, steps = 0, 0
    for _ in range(n):
        solved, s = run_episode(agent, depth, depth + 6, train=False)
        succ += solved
        steps += s
    return succ / n, steps / n


def train_curriculum(max_depth=8, episodes_per_check=500,
                      success_threshold=0.9, max_episodes_per_depth=20000):
    agent = QLearningAgent()
    depth_results = {}
    depth = 1
    t0 = time.time()
    while depth <= max_depth:
        episodes_this_depth = 0
        while episodes_this_depth < max_episodes_per_depth:
            for _ in range(episodes_per_check):
                run_episode(agent, depth, depth + 6, train=True)
            episodes_this_depth += episodes_per_check
            rate, steps = evaluate(agent, depth, n=150)
            elapsed = time.time() - t0
            print(f"[{elapsed:6.1f}s] depth={depth}  episodes={episodes_this_depth:6d}  "
                  f"success_rate={rate:5.1%}  avg_steps={steps:4.1f}  "
                  f"q_entries={len(agent.q)}")
            if rate >= success_threshold:
                break
        rate, steps = evaluate(agent, depth, n=300)
        depth_results[depth] = {"rate": rate, "avg_steps": steps}
        print(f"  -> depth {depth} final: success_rate={rate:.1%}  "
              f"avg_steps={steps:.1f}")
        depth += 1
    return agent, depth_results


def export_agent(agent, depth_results, max_depth, path="trained_agent.json"):
    """Collapse the (state,action)->value dict into {state: best_action_index}
    -- for a solving demo we only ever need the greedy action per state, and
    dropping the other 11 Q-values per state keeps the file small (state
    keys, not values, dominate file size)."""
    best_action = {}
    seen_states = {k for (k, a) in agent.q.keys()}
    for key in seen_states:
        qvals = agent._qvals(key)
        best_action[key] = max(range(N_ACTIONS), key=lambda a: qvals[a])

    payload = {
        "max_depth": max_depth,
        "n_actions": N_ACTIONS,
        "depth_results": depth_results,
        "policy": best_action,
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\nExported {len(best_action)} states to '{path}' ({size_mb:.1f} MB)")


def move_name(face, clockwise):
    return face if clockwise else face + "'"


def solve_once(agent, max_depth=8):
    """Scramble a fresh cube by max_depth random moves, then have the agent
    greedily attempt to solve it. Returns a dict in the same shape as
    solution.json: action indices and human-readable move strings for both
    the scramble and the attempted solution, plus whether it succeeded.
    """
    cube = Cube(2)
    scramble_moves = cube.scramble(max_depth)  # [(face, clockwise), ...]
    scramble_idx = [ACTIONS.index(m) for m in scramble_moves]

    solution_idx = []
    max_steps = max_depth + 6
    solved = cube.is_solved()
    stuck = False
    if not solved:
        key = js_state_key(cube)
        seen = {key}
        for _ in range(max_steps):
            action = agent.choose_action(key, greedy=True)
            face, cw = ACTIONS[action]
            cube.apply_move(face, cw)
            solution_idx.append(action)
            key = js_state_key(cube)
            solved = cube.is_solved()
            if solved:
                break
            if key in seen:
                stuck = True   # deterministic loop -- will never solve from here
                break
            seen.add(key)

    return {
        "depth": max_depth,
        "solved": solved,
        "stuck": stuck,
        "scramble": scramble_idx,
        "scramble_moves": [move_name(f, cw) for f, cw in scramble_moves],
        "solution": solution_idx,
        "solution_moves": [move_name(*ACTIONS[a]) for a in solution_idx],
    }


if __name__ == "__main__":
    MAX_DEPTH = 8
    print(f"Training 2x2 Q-learning agent up to depth {MAX_DEPTH} "
          f"(same defaults as q_learning_2x2.py)...\n")
    agent, depth_results = train_curriculum(max_depth=MAX_DEPTH)

    print("\nFinal results by depth:")
    for d, r in depth_results.items():
        print(f"  depth={d}: success_rate={r['rate']:.1%}  "
              f"avg_steps={r['avg_steps']:.1f}")

    export_agent(agent, depth_results, MAX_DEPTH)

    print("\nSample solve (sanity check -- should NOT be all the same move):")
    for _ in range(3):
        result = solve_once(agent, MAX_DEPTH)
        print(f"  solved={result['solved']}  scramble={result['scramble_moves']}  "
              f"solution={result['solution_moves']}")