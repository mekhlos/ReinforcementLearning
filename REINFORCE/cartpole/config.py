import rl
import numpy as np


class CartpoleAgent(rl.Agent):

    def observe(self):
        return self.environment.get_state().flatten()

    def get_next_action(self, state, network):
        action_probability_distribution = network.predict_one(state)

        action = np.random.choice(
            range(action_probability_distribution.shape[-1]),
            p=action_probability_distribution.flatten()
        )  # select action w.r.t the actions prob

        return action


def reward_function(state, action, new_state):
    pass
