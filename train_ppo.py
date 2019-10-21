import gym
import os
import tensorflow as tf  # 1.14.0
import numpy as np  # 1.16.4
import datetime
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy 
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Settings
env_id = 'CartPole-v1'
policy = MlpPolicy
multiprocessing = True
n_cpu = 4  # Number of processes to use
policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[32, 32])

# Hyperparameters
n_steps_per_update = 128  # The number of steps to run for each environment per update (batch_size = n_steps_per_update * n_cpu) (default: 128)
gamma = 0.99  # (0.99)
learning_rate = 0.001  # (2.5e-4)
n_mini_batches = 2   # The batch of size batch_size is divided into mini-batches of size batch_size / n_mini_batches, n_cpu % n_mini_batches == 0 (4)
n_updates = 1000

batch_size = n_steps_per_update * n_cpu
n_total_timesteps = batch_size * n_updates
eval_env = DummyVecEnv([lambda: gym.make(env_id)])


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed environment.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def callback(_locals, _globals):
    """
    Callback called at each update.
    :param _locals: (dict)
    :param _globals: (dict)
    """
    print('Update {} of {}'.format(_locals['update'], _locals['n_updates']))
    print(evaluate(eval_env, _locals['self']))
    return True


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")


def evaluate(env, model):
    """
    Returns average episodic rewards for given model.
    """
    episode_rewards = []
    for _ in range(10):
        reward_sum = 0
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _info = env.step(action)
            reward_sum += reward
        episode_rewards.append(reward_sum)
    return np.mean(episode_rewards)


def main():
    # Create log directory
    os.makedirs(env_id, exist_ok=True)
    log_dir = os.path.join(os.getcwd(), env_id, get_time())
    os.makedirs(log_dir)

    # Create environment
    if multiprocessing:
        env = SubprocVecEnv([make_env(env_id, i) for i in range(n_cpu)])
    else:
        env = DummyVecEnv([lambda: gym.make(env_id)])

    # Define the model
    model = PPO2(policy, env, 
        gamma=gamma, 
        n_steps=n_steps_per_update, 
        learning_rate=learning_rate, 
        nminibatches=n_mini_batches, 
        verbose=1, 
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs)

    # Train the agent
    model.learn(total_timesteps=n_total_timesteps, callback=callback)

    # After training, watch the agent walk
    obs = eval_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, _info = eval_env.step(action)
        if dones[0]:
            break
        eval_env.render()
    eval_env.close()

    # Evaluate agent
    print(evaluate(eval_env, model))

    # Save the model
    model.save(os.path.join(log_dir, 'ppo2_cartpole'))


if __name__ == "__main__":
    main()