"""Training-only joint navigation for the cooperative two-player, one-food task."""

from collections import deque

import numpy as np


def demo_actions(env):
    from lbforaging.foraging.environment import Action

    base = env.base.unwrapped
    food_cells = np.argwhere(base.field > 0)
    if len(base.players) != 2 or len(food_cells) != 1:
        raise ValueError("LBF demonstrations require two players and one food cell")

    food = tuple(int(coordinate) for coordinate in food_cells[0])
    shape = base.field.shape
    start = tuple(tuple(player.position) for player in base.players)
    moves = {
        Action.NORTH: (-1, 0),
        Action.SOUTH: (1, 0),
        Action.WEST: (0, -1),
        Action.EAST: (0, 1),
    }
    # Move one agent at a time to avoid collisions, then LOAD on the same step.
    frontier = deque([start])
    parent = {start: None}
    while frontier:
        pair = frontier.popleft()
        if all(abs(row - food[0]) + abs(col - food[1]) == 1 for row, col in pair):
            path = [(Action.LOAD, Action.LOAD)]
            while parent[pair] is not None:
                pair, action = parent[pair]
                path.append(action)
            return (
                dict(zip(env.possible_agents, (action.value for action in joint)))
                for joint in reversed(path)
            )
        for index, (row, col) in enumerate(pair):
            for action, (dr, dc) in moves.items():
                target = (row + dr, col + dc)
                if target in (food, pair[1 - index]) or any(
                    not 0 <= coordinate < limit for coordinate, limit in zip(target, shape)
                ):
                    continue
                candidate = (target, pair[1]) if index == 0 else (pair[0], target)
                if candidate in parent:
                    continue
                joint = (action, Action.NONE) if index == 0 else (Action.NONE, action)
                parent[candidate] = pair, joint
                frontier.append(candidate)

    raise ValueError("No legal LBF demonstration path reaches the food")
