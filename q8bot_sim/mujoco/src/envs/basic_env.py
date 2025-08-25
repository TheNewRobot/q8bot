import numpy as np

from mujoco_playground._src import mjx_env
from configs.base_config import Q8BotConfig
import jax
import jax.numpy as jp
from mujoco import mjx

import os
from mujoco_playground._src.locomotion.spot.base import SpotEnv
from mujoco_playground._src.locomotion.spot import spot_constants as consts
from mujoco_playground._src.locomotion.spot.getup import default_config
from pathlib import Path

class Q8BotEnv(mjx_env.MjxEnv):
    """Tiny quadruped environment class"""
    def __init__(self, task = "standing", config = 'None'):
        # TODO: Load Q8Bot model when available
        # model_path = "models/q8bot.xml"
        raise NotImplementedError("Q8Bot URDF not available. Use SpotPlaceholderEnv.")

class SpotPlaceholderEnv(SpotEnv):
    """Placeholder class for Q8BotEnv Development"""
    
    def __init__(self, task="standing", config=None):
        self.task = task
        self.q8_config = config or Q8BotConfig()
        
        xml_path = Path(consts.FULL_FLAT_TERRAIN_XML).resolve()
        original_cwd = os.getcwd()
        
        os.chdir(xml_path.parent)
        try:
            super().__init__(xml_path=str(xml_path), config=default_config())
        finally:
            os.chdir(original_cwd)
        self._post_init()
    
    def _post_init(self):
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = self._mj_model.keyframe("home").qpos[7:]
        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
        self._up_vec = jp.array([0.0, 0.0, 1.0])
        self._step_count = 0

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, noise_rng = jax.random.split(rng, 2)
        
        # Initialize at default pose
        qpos = self._init_q
        
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=jp.zeros(self.mjx_model.nv),
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)
        data = data.replace(time=0.0)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "step_count": 0,
        }

        metrics = {
            "reward/height": jp.zeros(()),
            "reward/upright": jp.zeros(()),
            "reward/stability": jp.zeros(()),
            "reward/forward": jp.zeros(()),
        }

        obs = self._get_obs(data, info, noise_rng)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, noise_rng = jax.random.split(state.info["rng"], 2)
        
        # Apply action with scaling
        motor_targets = state.data.qpos[7:] + action * self.q8_config.action_scale
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
        
        # Physics step
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )
        
        # Get observation
        obs = self._get_obs(data, state.info, noise_rng)
        
        # Compute reward
        reward = self._compute_reward(data, action, state.info)
        
        # Check termination conditions
        done = jp.float32(data.qpos[2] < self.q8_config.fall_height_threshold)
        step_count = state.info["step_count"] + 1
        done = jp.logical_or(done, step_count >= self.q8_config.max_episode_steps)
        
        # Update info
        new_info = state.info.copy()
        new_info["last_act"] = action
        new_info["rng"] = rng
        new_info["step_count"] = step_count
        
        # Update metrics (for monitoring)
        new_metrics = {
            "reward/total": reward,
        }
        
        return state.replace(
            data=data, obs=obs, reward=reward, done=done, 
            metrics=new_metrics, info=new_info
        )

    def _get_obs(self, data: mjx.Data, info: dict, rng: jax.Array) -> jax.Array:
        """Simple observation: gravity + joint angles"""
        gravity = self.get_gravity(data)
        joint_angles = data.qpos[7:]
        
        return jp.concatenate([
            gravity,  # 3
            joint_angles - self._default_pose,  # 12
        ])

    def _compute_reward(self, data: mjx.Data, action: jax.Array, info: dict) -> jax.Array:
        """Simple reward: stay upright at target height"""
        # Height reward
        height_error = jp.abs(data.qpos[2] - self.q8_config.body_height)
        height_reward = jp.exp(-height_error * 10.0)
        
        # Upright reward
        gravity = self.get_gravity(data)
        upright_error = jp.sum(jp.square(self._up_vec - gravity))
        upright_reward = jp.exp(-upright_error * 5.0)
        
        return height_reward + upright_reward


if __name__ == '__main__':
    config = Q8BotConfig(body_height=0.3, kp_joint=300.0)
    env = SpotPlaceholderEnv(task="standing", config=config)
    rng = jax.random.PRNGKey(0)
    
    state = env.reset(rng)
    
    for i in range(100):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.normal(action_rng, (env.action_size,)) * 0.1
        
        state = env.step(state, action)
        
        print(f"Step {i}: reward={state.reward:.3f}, done={state.done}, height={state.data.qpos[2]:.3f}")
        
        if state.done:
            print("Episode terminated, resetting...")
            state = env.reset(rng)