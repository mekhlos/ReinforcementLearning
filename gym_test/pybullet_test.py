import pybullet
import gym

env = gym.make('RacecarBulletEnv-v0')
# env = gym.make('Ant-v2')
# env = gym.make('BipedalWalker-v2')
env.reset()

while True:
    env.step(env.action_space.sample())
    env.render()
