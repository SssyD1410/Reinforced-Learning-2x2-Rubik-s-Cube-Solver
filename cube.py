"""
cube.py -- A from-scratch Rubik's Cube environment for RL experiments.

Supports the 2x2 (corners only) and 3x3 (corners + edges + centers) cubes,
using a 3D-coordinate cubie representation rather than a hand-copied
permutation table. Each small cubie lives at an integer coordinate
(x, y, z) with entries in {-1, 0, 1}, and carries a dict of
{outward_direction: face_color} stickers. A face turn is just a 90-degree
rotation, about one axis, applied to every cubie in that layer -- both its
position and the directions its stickers point. This makes the logic easy
to verify: turning a face 4 times, or a move followed by its inverse, must
always return the cube to its starting state (see the self-tests at the
bottom of this file).
"""
from __future__ import annotations
import random
from itertools import product
import numpy as np

FACE_NORMALS = {
    'U': (0, 1, 0),
    'D': (0, -1, 0),
    'R': (1, 0, 0),
    'L': (-1, 0, 0),
    'F': (0, 0, 1),
    'B': (0, 0, -1),
}

DIRECTION_TO_FACE = {v: k for k, v in FACE_NORMALS.items()}

AXIS_OF_FACE = {'U': 'y', 'D': 'y', 'R': 'x', 'L': 'x', 'F': 'z', 'B': 'z'}
LAYER_VALUE_OF_FACE = {'U': 1, 'D': -1, 'R': 1, 'L': -1, 'F': 1, 'B': -1}
AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def _rotate_vec(v, axis, sign):
    """Rotate by 90*sign degrees about `axis`.v is the triple being rotated."""
    x, y, z = v
    if axis == 'x':
        return (x, -sign * z, sign * y)
    if axis == 'y':
        return (sign * z, y, -sign * x)
    if axis == 'z':
        return (-sign * y, sign * x, z)
    raise ValueError(axis)


class Cubie:
    __slots__ = ('pos', 'stickers')

    def __init__(self, pos, stickers):
        self.pos = pos                    # (x, y, z)
        self.stickers = dict(stickers)    # {direction: face_letter}

    def rotate(self, axis, sign):
        self.pos = _rotate_vec(self.pos, axis, sign)
        new_stickers = {}
        for d, color in self.stickers.items():
            new_direction = _rotate_vec(d, axis, sign)
            new_stickers[new_direction] = color
        self.stickers = new_stickers

    def clone(self):
        return Cubie(self.pos, dict(self.stickers))


class Cube:
    """An N x N x N cube. N is 2 or 3"""

    FACES = ['U', 'D', 'R', 'L', 'F', 'B']
    # 12 actions, 6 faces x 2 directions, clockwise is true
    ACTIONS = [(f, cw) for f in FACES for cw in (True, False)]  

    def __init__(self, n: int = 3):
        assert n in (2, 3), "this simple implementation supports N=2 or N=3"
        self.n = n
        coord_vals = [-1, 1] if n == 2 else [-1, 0, 1]
        self.cubies = []
        for pos in product(coord_vals, repeat=3):
            if n == 3 and pos.count(0) == 3:
                continue  # invisible core piece, nothing to track
            stickers = {}
            for direction in FACE_NORMALS.values():
                axis = next(
                    i for i, c in enumerate(direction) 
                    if c != 0
                )
                if pos[axis] == direction[axis]:
                    stickers[direction] = DIRECTION_TO_FACE[direction]
            self.cubies.append(Cubie(pos, stickers))

    # ---- core mechanics -------------------------------------------------

    def clone(self) -> "Cube":
        c = Cube.__new__(Cube)
        c.n = self.n
        c.cubies = [cb.clone() for cb in self.cubies]
        return c

    def apply_move(self, face: str, clockwise: bool = True):
        axis = AXIS_OF_FACE[face]
        layer_val = LAYER_VALUE_OF_FACE[face]
        axis_idx = AXIS_INDEX[axis]
        # 'clockwise' always means clockwise viewed from outside that face,
        # regardless of whether the layer sits on the + or - side.
        eff_sign = (1 if clockwise else -1) * layer_val
        for cubie in self.cubies:
            if cubie.pos[axis_idx] == layer_val:
                cubie.rotate(axis, eff_sign)

    def apply_action(self, action_idx: int):
        face, cw = self.ACTIONS[action_idx]
        self.apply_move(face, cw)

    def scramble(self, n_moves: int, rng: random.Random = None):
        rng = rng or random
        moves = []
        for _ in range(n_moves):
            face = rng.choice(self.FACES)
            cw = rng.choice([True, False])
            self.apply_move(face, cw)
            moves.append((face, cw))
        return moves

    # ---- reading state ----------------------------------------------------

    def is_solved(self) -> bool:
        by_face = {f: [] for f in self.FACES}
        for cubie in self.cubies:
            for direction, color in cubie.stickers.items():
                by_face[DIRECTION_TO_FACE[direction]].append(color)
        return all(len(set(colors)) == 1 for colors in by_face.values())

    def state_key(self):
        """Hashable state. can be used as a dict/Q-table key."""
        parts = []
        for cubie in sorted(self.cubies, key=lambda c: c.pos):
            parts.append((cubie.pos, tuple(sorted(cubie.stickers.items()))))
        return tuple(parts)

    def state_vector(self) -> np.ndarray:
        """Flat one-hot float vector -- usable as a neural net input."""
        color_ids = {f: i for i, f in enumerate(self.FACES)}
        vec = []
        for cubie in sorted(self.cubies, key=lambda c: c.pos):
            for _, color in sorted(cubie.stickers.items()):
                one_hot = [0.0] * 6
                one_hot[color_ids[color]] = 1.0
                vec.extend(one_hot)
        return np.array(vec, dtype=np.float32)

    @property
    def state_size(self) -> int:
        return len(self.state_vector())


# ---------------------------------------------------------------------------
# Self-tests: run `python cube.py` to sanity-check the move logic.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for n in (2, 3):
        print(f"Testing N={n} cube...")
        c = Cube(n)
        assert c.is_solved()

        # Turning any face 4 times must return to the identical state.
        for face in Cube.FACES:
            test = c.clone()
            for _ in range(4):
                test.apply_move(face, True)
            assert test.state_key() == c.state_key(), f"{face} x4 failed"

        # A move followed by its inverse must cancel out.
        for face in Cube.FACES:
            test = c.clone()
            test.apply_move(face, True)
            test.apply_move(face, False)
            assert test.state_key() == c.state_key(), f"{face}/{face}' failed"

        # Scramble then apply the exact inverse sequence -> solved again.
        test = c.clone()
        moves = test.scramble(25)
        for face, cw in reversed(moves):
            test.apply_move(face, not cw)
        assert test.is_solved(), "scramble + inverse sequence did not solve"

        # A generic scramble should (almost always) NOT be solved.
        test = c.clone()
        test.scramble(10)
        assert not test.is_solved()

        print(f"  state vector length: {c.state_size}")
        print(f"  all tests passed for N={n}")

    print("cube.py: all self-tests passed.")
