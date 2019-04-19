import numpy as np

from environments.MyWorld1.behaviours import behaviour_interface


class RandomBehaviour1(behaviour_interface.BehaviourInterface):

    def __init__(self, n_sides, side_length):
        super().__init__()

        angles = [2 * np.pi * i / n_sides for i in range(n_sides)]
        self.cosv = side_length * np.cos(angles)
        self.sinv = side_length * np.sin(angles)

        self.current_i = 0

    def get_delta_position(self):
        dx, dy = self.cosv[self.current_i], self.sinv[self.current_i]
        self.current_i = (len(self.cosv) + self.current_i + ((np.random.rand() > 0.5) * 2 - 1)) % len(self.cosv)

        return dx, dy


class RandomBehaviour2(behaviour_interface.BehaviourInterface):

    def __init__(self, n_sides, side_length, width, height):
        super().__init__()
        circle_radius = 30

        angles = [2 * np.pi * i / n_sides for i in range(n_sides)]
        self.cosv = side_length * np.cos(angles)
        self.sinv = side_length * np.sin(angles)

        self.radius = (n_sides * side_length) / (2 * np.pi) * np.cos(np.pi / n_sides)
        self.min_x = self.radius + circle_radius
        self.min_y = self.radius + circle_radius
        self.max_x = width - self.radius * 2 - circle_radius * 2
        self.max_y = height - self.radius * 2 - circle_radius * 2

        self.direction = 1
        self.angle_index = 0

    @staticmethod
    def get_random_direction():
        return (np.random.rand() > 0.5) * 2 - 1

    def get_new_position(self, old_position):
        new_direction = self.get_random_direction()
        new_angle_index = (len(self.cosv) + self.angle_index + new_direction) % len(self.cosv)
        new_x = old_position[0] + self.cosv[new_angle_index]
        new_y = old_position[1] + self.sinv[new_angle_index]

        if self.min_x < new_x < self.max_x and self.min_y < new_y < self.max_y:
            self.direction = new_direction
            self.angle_index = new_angle_index
        else:
            self.angle_index = (len(self.cosv) + self.angle_index + self.direction) % len(self.cosv)
            new_x = old_position[0] + self.cosv[self.angle_index]
            new_y = old_position[1] + self.sinv[self.angle_index]

        return new_x, new_y
