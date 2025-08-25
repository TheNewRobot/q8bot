import numpy as np
from dataclasses import dataclass

@dataclass
class Q8BotConfig:
    body_height: float = 0.25
    leg_length: float = 0.15
    body_width: float = 0.2
    kp_joint: float = 200.0
    kd_joint: float = 20.0
    kp_balance: float = 100.0
    # Reward weights
    height_reward_weight: float = 1.0
    upright_reward_weight: float = 1.0
    stability_reward_weight: float = 0.1
    fall_penalty: float = -10.0
    forward_reward_weight: float = 1.0
    # Environment settings
    dt: float = 0.01
    max_episode_steps: int = 1000
    fall_height_threshold: float = 0.1
    action_scale: float = 0.6
    # Standing Pose
    default_joint_angles: np.ndarray = None

    def __post_init__(self):
        if self.default_joint_angles is None:
            self.default_joint_angles = np.array([
                0.0, 0.8, -1.6,  # FL
                0.0, 0.8, -1.6,  # FR
                0.0, 0.8, -1.6,  # RL
                0.0, 0.8, -1.6   # RR
            ])