import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import time
from src.envs.basic_env import SpotPlaceholderEnv, Q8BotConfig


class SimplePositionController:
    """Simple PD controller to maintain default standing pose"""
    
    def __init__(self, kp=10.0, kd=1.0):
        self.kp = kp  # Position gain
        self.kd = kd  # Velocity gain
        
    def get_action(self, observation, target_pose=None):
        """Compute control action to maintain target pose"""
        # Extract joint angles from observation (skip gravity vector)
        joint_angles_error = observation[3:15]  # 12 joint angles relative to default
        
        if target_pose is not None:
            # Override with custom target
            joint_angles_error = observation[3:15] - target_pose
        
        # Simple PD control: drive joint errors to zero
        action = -self.kp * joint_angles_error
        return jp.clip(action, -1.0, 1.0)


def run_controller():
    """Run position controller with text output only"""
    config = Q8BotConfig(body_height=0.25, kp_joint=300.0)
    env = SpotPlaceholderEnv(task="standing", config=config)
    controller = SimplePositionController(kp=5.0, kd=0.5)
    
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    print("Running position controller...")
    print(f"Target height: {config.body_height}")
    
    for i in range(500):
        action = controller.get_action(state.obs)
        state = env.step(state, action)
        
        if i % 50 == 0:
            height = state.data.qpos[2]
            print(f"Step {i}: reward={state.reward:.3f}, height={height:.3f}")
        
        if state.done:
            print("Robot fell, resetting...")
            state = env.reset(rng)


def run_controller_with_rendering():
    """Run position controller with 3D visualization"""
    config = Q8BotConfig(body_height=0.25, kp_joint=300.0)
    env = SpotPlaceholderEnv(task="standing", config=config)
    step_fn = jax.jit(env.step)
    controller = SimplePositionController(kp=5.0, kd=0.5)
    
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    print("Starting visual controller...")
    print(f"Target height: {config.body_height}")
    print("Press Ctrl+C to stop")
    
    # Convert JAX data to numpy for MuJoCo viewer
    mj_data = mujoco.MjData(env.mj_model)
    
    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        for i in range(5000):
            # Get control action
            action = controller.get_action(state.obs)
            
            # Step environment
            state = step_fn(state, action)
            
            # Copy state to MuJoCo data for rendering
            mj_data.qpos[:] = state.data.qpos
            mj_data.qvel[:] = state.data.qvel
            mj_data.time = state.data.time
            
            # Forward kinematics for rendering
            mujoco.mj_forward(env.mj_model, mj_data)
            
            # Update viewer
            viewer.sync()
            
            # Print status
            if i % 1 == 0:
                height = state.data.qpos[2]
                print(f"Step {i}: reward={state.reward:.3f}, height={height:.3f}")
            
            if state.done:
                print("Robot fell, resetting...")
                state = env.reset(rng)
            # breakpoint()

if __name__ == '__main__':
    # Choose which version to run:
    # run_controller()  # Text only
    run_controller_with_rendering()  # With 3D visualization