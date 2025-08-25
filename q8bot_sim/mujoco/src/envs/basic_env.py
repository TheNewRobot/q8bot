import numpy

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spot.base import SpotEnv
from configs import Q8BotConfig

class Q8BotEnv(mjx_env.MjxEnv):
    """Tiny quadruped environment class"""
    def __init__(self, task = "standing", config = 'None'):
        # TODO: Load Q8Bot model when available
        # model_path = "models/q8bot.xml"
        raise NotImplementedError("Q8Bot URDF not available. Use SpotPlaceholderEnv.")

class SpotPlaceholderEnv(SpotEnv):
    """Placeholder class for Q8BotEnv Development"""
    def __init__(self, task="standing", config=None):
        super().__init__()
        self.task = task
        self.config = config or Q8BotConfig()

if __name__ == '__main__':
    config = Q8BotConfig(body_height = 0.3, kp_joint = 300.0)
    env = SpotPlaceholderEnv(task="standing", config=config)
