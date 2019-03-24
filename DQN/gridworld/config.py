import rl


class GridworldAgent(rl.Agent):

    def observe(self):
        environment_state = self.environment.get_state()
        state = environment_state[[0, 1, 3]]
        return state.flatten()

    def get_next_action(self, state, network):
        q_values = network.predict_one(state)
        return q_values.argmax()


def reward_function(state, action, new_state):
    pass
