# Reinforcement Learning Environments for Surgical Robotics Challenge
This project proposes a reinforcement learning (RL) interface for robotic suturing simulation environment based on Gymnasium and Stable baseline3.


## Prerequiste
This section introduces the necessary configuration you need.
### System Requirement
* Ubuntu 20.04.

### Installation
* Install the [surgical robotics challenge environment](https://github.com/surgical-robotics-ai/surgical_robotics_challenge) as well as the AMBF and ROS prerequisites in the link. It provides simulation environment for suturing phantom combined with da Vinci surgical system.
```
git clone https://github.com/surgical-robotics-ai/surgical_robotics_challenge
```
* Install Gymnasium: [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) is a branch and updated version of OpenAI Gym. It provides standard API for the communication between the simulated environment and learning algorithms.
```
pip install gymnasium
```

* Configure the [Pytorch](https://pytorch.org/) and [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (if equipped with NVIDIA card) based on your hardware.

* Install [Stable Baseline3](https://github.com/DLR-RM/stable-baselines3) (SB3): SB3 is an open-source Python library providing implementations of state-of-the-art RL algorithms. In this project, it is used to interaction with Gymnasium environment and offering interface for training, evaluating, and testing RL models.
```
pip install stable-baselines3
```

### Installation Verification
* Try to run the code below. This script will create a embedded RL environment from Gymnasium and train it using the PPO algorithm from Stable Baselines3. If everything is set up correctly, the script should run without any errors.
``` python
import gymnasium
import stable_baselines3

env = gymnasium.make('CartPole-v1')
model = stable_baselines3.PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
```

## RL Training
This section introduce the basic procedure for model training with defined Gymnasium environment.

### Run the SRC Environment
Make sure ROS and SRC is running before moving forward to the following steps. You can simply run the following command or refer to this [link](https://github.com/surgical-robotics-ai/surgical_robotics_challenge) for details.

```
roscore
```
```
~/ambf/bin/lin-x86_64/ambf_simulator --launch_file ~/ambf/surgical_robotics_challenge/launch.yaml -l 0,1,3,4,13,14 -p 200 -t 1 --override_max_comm_freq 120
```

### Register the Gymnasium Environment
```python
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from SRC_test import SRC_test
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
gym.envs.register(id="Training_ppo_first", entry_point=SRC_test, max_episode_steps=5000)
env = gym.make("Training_ppo_first", render_mode="human")
```

### Initialize and Train the Model
Here is an example of model with Proximal Policy Optimization (PPO) algorithm.
```python
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./First_version/",)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./First_version/Model_temp', name_prefix='SRC')
model.learn(total_timesteps=int(1000000), progress_bar=True,callback=checkpoint_callback,)
model.save("SRC")
```

### Load the Model
```python
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./First_version/",)
model_path = "./First_version/Model_temp/SRC_10000_steps.zip"
model = PPO.load(model_path)
model.set_env(env=env)
```

### Test the Model Prediction
```python
obs,info = env.reset()
print(obs)
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, terminated,truncated, info = env.step(action)
    print(info)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
``` 
The following video demonstrates a training simulation specifically designed for needle grasping task.

[demo](https://github.com/JinnWu686/SRC_gym/assets/147576462/81b13709-bba1-4802-8756-26bc2183cc09)


## Documentation
This section describes the current settings of the RL environment, though it is flexible for further tuning and improvement.

### 'SRC_test' Class:
This is the main environment class that extends gym.Env. It defines the action space, observation space, reward functions, and contains the core logic for the simulation environment.

### Initialization:
'__init__' function initializes the environment, setting up the action and observation spaces, including their upper and lower boundaries. 

For instance, the action space (7*1 array) in this case represents 6D pose of PSM end-effector and state of gripper. 
Discrete action space is used in this case defined as follows, each component will be assign to a value from [0,1,2]:
```python
action_space = spaces.MultiDiscrete([3,3,3,3,3,3,3])
```
The corresponding boundaries are as follows, to ensure each joint won't exceed their physical limit.
```python
self.action_lims_low = np.array([-0.15, -0.15, -0.25, np.deg2rad(-350), np.deg2rad(-85), np.deg2rad(-260), 0],dtype=np.float32)
self.action_lims_high = np.array([0.15, 0.15, 0.05, np.deg2rad(-10), np.deg2rad(85), np.deg2rad(260), 1],dtype=np.float32)
```

It also builds connection with simulator components like PSM and ECM with code like below:
```python
self.simulation_manager = SimulationManager('src_client')
self.world_handle = self.simulation_manager.get_world_handle()
self.scene = Scene(self.simulation_manager)
self.psm1 = PSM(self.simulation_manager, 'psm1',add_joint_errors=False)
self.psm2 = PSM(self.simulation_manager, 'psm2',add_joint_errors=False)
self.ecm = ECM(self.simulation_manager, 'CameraFrame')
```

### Reset:
'__reset__' function is also a necessary component in Gymnasium environment, It resets the environment to its initial state at the beginning of each episode. In our case,
it basically involves recovering all joints of robot arm and view of camera to their original positions. Code belows shows the setting initial positions of PSMs.
```
self.psm1.servo_jp([0.01102378,-0.01772329,0.10328061,-0.00375491,-0.03087782,0.21085757])
self.psm2.servo_jp([-0.00575528,-0.0083593,0.10328144,0.00249357,-0.01751206,-0.13059567])
```


### Step:
'__step__' method is used to execute one timestep within the environment using the given action. At each timestep, an updated action_space is generated based on the current state. The code below shows how PSM is controlled by the command.
```python
action_discount = np.clip(0.1,0.4,10/3*(self.obs.dist-0.01)+0.1)
action_step = (action-1)*np.array([0.001,0.001,0.001,0.02,0.02,0.02,0.2])*action_discount
self.psm2_goal = np.clip(current+action_step, self.action_lims_low[0:7], self.action_lims_high[0:7])
self.psm2_step(self.psm2_goal)
```
The function also returns the termination and truncation state of the episode. For instance, in grasping process, the termination is true when disance between the needle and gripper is smaller than a threshold and gripper in a appropriate orientation. Meanwhile, truncation is activated when maximum timesteps is reached. Both termination and truncation will lead to the end of the current episode. 

### Reward:
'__reward__' function computes the reward for the current action. The formula here shows how reward functions are designed for needle-grasping task.

$$
\begin{aligned}
& \text{Priority}_{dist} = \begin{cases}
120 & \text{if timestep} < 0.5 \cdot \text{max}\_\text{timestep} \\
180 & \text{if timestep} \geq 0.5 \cdot \text{max}\_\text{timestep}
\end{cases}
\end{aligned}
$$

$$
\begin{aligned}
& \text{Priority}_{angle} = \begin{cases}
12 & \text{if timestep} < 0.5 \cdot \text{max}\_\text{timestep} \\
6 & \text{if timestep} \geq 0.5 \cdot \text{max}\_\text{timestep}
\end{cases}
\end{aligned}
$$

$$
\text{Reward}\_{dist} = \begin{cases}
\text{Priority}\_{dist} \cdot \delta\_{dist} - \text{dist}/15 & \text{if } \delta\_{dist} < 0 \\
\text{Priority}\_{dist} \cdot 2 \cdot \delta\_{dist} - \text{dist}/15 & \text{if } \delta\_{dist} \geq 0
\end{cases}
$$

$$
\text{Reward}\_{angle} = \begin{cases}
\text{Priority}\_{angle} \cdot {\delta\_{angle} - \text{angle}}/500 & \text{if } \delta\_{angle} < 0 \\
\text{Priority}\_{angle} \cdot {2 \cdot \delta\_{angle} - \text{angle}}/500 & \text{if } \delta\_{angle} \geq 0
\end{cases}
$$

$$
\text{Reward}\_{time} = -0.004
$$

$$
\text{Reward}\_{success} = 60 \text{   if success}
$$

$$
\text{Reward}\_{range} = -80 \text{   if out of range}
$$

$$
\text{Reward}\_{grasp} = \text{Reward}\_{dist} + \text{Reward}\_{angle} + \text{Reward}\_{time} + \text{Reward}\_{success} + \text{Reward}\_{range}
$$

Here dist and angle represent the end-effector translational distance and orientation error between the current and goal state.
