from DQN.trainer import DQNTeacher


class DQNTeacherWithTable(DQNTeacher):
    def __init__(self, agent, replay_memory, env, exploration_helper, settings, q_table, save_path):
        self.q_table = q_table
        super().__init__(agent, replay_memory, env, self.q_table, exploration_helper, settings, save_path)

    def replay(self):
        alpha = 1 / self.settings.BATCH_SIZE / 100
        batch = self.replay_memory.sample(self.settings.BATCH_SIZE)

        for i, (s, a, r, s2, t) in enumerate(zip(*batch)):
            old_q = self.q_table[s, a]
            new_q = self.q_table.get_q_values_for_state(s2)
            if t:
                target = r
            else:
                target = r + self.settings.DISCOUNT_FACTOR * new_q.max()

            update = (1 - alpha) * old_q + alpha * target

            self.q_table.update(s, a, update)

        return 0


if __name__ == '__main__':
    pass
