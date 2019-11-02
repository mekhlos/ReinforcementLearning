import pybullet
import pybullet_envs
import gym
import time

# import pybullet_envs.bullet.racecarGymEnv as e

# env = e.RacecarGymEnv(renders=True)
# env = gym.make('HalfCheetahBulletEnv-v0', render=True)
env = gym.make('AntBulletEnv-v0', render=True)
env.reset()

while True:
    env.step(env.action_space.sample())
    env.render()
    time.sleep(.1)
