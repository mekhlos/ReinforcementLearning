import rl


class CartpoleAgent(rl.Agent):

    def observe(self):
        return self.environment.get_state().flatten()

    def get_next_action(self, state, network):
        q_values = network.predict_one(state)
        return q_values.argmax()


def reward_function(state, action, new_state):
    pass
