import tensorflow as tf
import gym
import rl
import numpy as np

from REINFORCE.cartpole import network as network_module
from environments import env_wrapper


def test(env, agent, network_interface):
    env.reset()
    rewards = []

    for episode_ix in range(10):
        env.reset()
        state = agent.observe()

        total_rewards = 0
        print('****************************************************')
        print(f'EPISODE {episode_ix}')

        while True:
            env.display()
            # Choose action a, remember we're not in a deterministic environment, we output probabilities.
            action_probability_distribution = network_interface.predict_one(state)

            # print(action_probability_distribution)
            action = np.random.choice(
                range(action_probability_distribution.shape[1]),
                p=action_probability_distribution.ravel()
            )  # select action w.r.t the actions prob

            _, reward, is_terminal, info = env.update(action)
            new_state = agent.observe().copy()

            total_rewards += reward

            if is_terminal:
                rewards.append(total_rewards)
                print('Score', total_rewards)
                break

            state = new_state

    env.close()
    print(f'Score over time: {sum(rewards) / 10}')


class Settings:
    INPUT_DIM = 4
    N_ACTIONS = 2
    DISCOUNT_FACTOR = 0.95


if __name__ == '__main__':
    settings = Settings()

    env = env_wrapper.EnvWrapper(gym.make('CartPole-v0'))
    agent = rl.Agent('test1', env, observe_function=env.observe_f)

    network = network_module.Network(settings.INPUT_DIM, settings.N_ACTIONS)
    network_interface = network_module.NetworkInterface(tf.Session(), network)
    # Load the model
    network_interface.restore(path='./models/model.ckpt')

    test(env, agent, network_interface)
