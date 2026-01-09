import minigrid
import gymnasium as gym
import numpy as np


def minigrid_state_processor(obs, env: gym.Env):
    return (env.unwrapped.agent_pos[0],
            env.unwrapped.agent_pos[1],
            env.unwrapped.agent_dir,
            env.unwrapped.goal_pos[0],
            env.unwrapped.goal_pos[1])


class FourRoomsWrapper(gym.ActionWrapper):
    def __init__(self, env, actions_map, agent_start=(7, 7), goal_pos=(2, 2), fixed_pos=True):
        super().__init__(env)
        self.actions_map = actions_map
        self.action_space = gym.spaces.Discrete(len(actions_map))
        self.agent_start = np.array(agent_start)
        self.goal_pos = np.array(goal_pos)
        self.fixed_pos = fixed_pos

    def action(self, action_idx):
        return self.actions_map[action_idx]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.fixed_pos:
            self.unwrapped.agent_pos = self.agent_start
            self.unwrapped.agent_dir = 0

            self.unwrapped.goal_pos = self.goal_pos

            self.unwrapped.grid.set(self.agent_start[0], self.agent_start[1], None)
            self.unwrapped.grid.set(self.goal_pos[0], self.goal_pos[1], minigrid.core.world_object.Goal())

        return obs, info
