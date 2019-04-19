class BehaviourInterface:
    def __init__(self):
        pass

    def get_delta_position(self):
        raise NotImplementedError('Please implement me, idiot!')

    def get_new_position(self, old_position):
        raise NotImplementedError('Please implement me, idiot!')

    def get_new_state(self, old_state):
        raise NotImplementedError('Please implement me, idiot!')
