import minigrid
import gymnasium as gym


def minigrid_state_processor(obs, env: gym.Env):
    return (env.unwrapped.agent_pos[0],
            env.unwrapped.agent_pos[1],
            env.unwrapped.agent_dir)


class ResrtrictedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, actions_map):
        super().__init__(env)
        self.actions_map = actions_map
        self.action_space = gym.spaces.Discrete(len(actions_map))

    def action(self, action_idx):
        return self.actions_map[action_idx]
