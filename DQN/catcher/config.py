import rl


class CatcherAgent(rl.Agent):

    def observe(self):
        state = self.environment.get_state()
        subsampled_state = state
        x = (subsampled_state.flatten() > 0).astype(int)
        return x

    def get_next_action(self, state, network):
        q_values = network.predict_one(state)
        return q_values.argmax()


def reward_function(state, action, new_state):
    pass
