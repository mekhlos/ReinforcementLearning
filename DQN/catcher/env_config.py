import numpy as np


class Config2:
    width = 600
    height = 800

    # Size of break-out blocks
    block_width = 23
    block_height = 15

    ball_width = 10
    ball_height = 10

    player_width = 75
    player_height = 15

    font_size = 36

    action_scale = 5

    n_block_rows = 5
    n_block_columns = 32

    block_start_from_top = 80
    block_padding = 2

    fps = 120
    text_start_from_top = 300
    player_x = 0


class Config1:
    width = 200
    height = 200

    # Size of break-out blocks
    block_width = 19
    block_height = 19

    ball_width = 10
    ball_height = 10

    ball_x = 100
    ball_y = 100

    player_width = 40
    player_height = 8

    font_size = 36

    action_scale = 4

    n_block_rows = 2
    n_block_columns = 10

    block_start_from_top = 20
    block_padding = 1

    fps = 20
    text_start_from_top = 50

    n_lives = 3

    @staticmethod
    def player_x():
        # return np.random.randint(0, Config1.width - Config1.player_width)
        return Config1.width / 2
