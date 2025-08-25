# Q8Bot Playground

This simulation suite is meant to help the software development of Q8bot.

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.x (optional for accelerated GPU cbackend)

### Setup
```bash
# Clone and environment
git clone git@github.com:EricYufengWu/q8bot.git
cd q8bot/q8bot_simulation/mujoco
python -m venv q8bot_sim
source q8bot_sim/bin/activate

# Install dependencies
pip install playground 
pip install "jax[cuda12]" numpy

# Fix JAX CUDA (if needed), whenver you have the problem that JAX doesn't find some libraries
unset LD_LIBRARY_PATH
```
**Note:** Check the instructions from the [main repo](https://github.com/google-deepmind/mujoco_playground/) if you want to install playground from source

### Dependencies
- `mujoco >= 3.0` - Physics simulation
- `jax[cuda12]` - GPU acceleration
- `numpy` - Numerical operations
- `mujoco-mjx` - JAX-based MuJoCo

## Project Structure

```
src/
├── environments/    # Q8Bot environments
├── controllers/     # Standing/balance control
└── utils/          # Helper functions
```

## Current Status
- ✅ Standing controller with a Robot placeholder
- ✅ Basic environment structure
- ⏳ Q8Bot URDF/MJF integration (pending)
- ⏳ Make the Q8bot environment pip installable 