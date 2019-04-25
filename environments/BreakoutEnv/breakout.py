"""
 Sample Breakout Game

 Sample Python/Pygame Programs
 Simpson College Computer Science
 http://programarcadegames.com/
 http://simpson.edu/computer-science/
"""

# --- Import libraries used for this program

import math
import pygame
import numpy as np

from DQN.breakout.env_config import Config1 as config

# Define some colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)

# Size of break-out blocks
block_width = config.block_width
block_height = config.block_height


class Block(pygame.sprite.Sprite):
    """This class represents each block that will get knocked out by the ball
    It derives from the "Sprite" class in Pygame """

    def __init__(self, color, x, y):
        """ Constructor. Pass in the color of the block,
            and its x and y position. """

        # Call the parent class (Sprite) constructor
        super().__init__()

        # Create the image of the block of appropriate size
        # The width and height are sent as a list for the first parameter.
        self.image = pygame.Surface([block_width, block_height])

        # Fill the image with the appropriate color
        self.image.fill(color)

        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()

        # Move the top left of the rectangle to x,y.
        # This is where our block will appear..
        self.rect.x = x
        self.rect.y = y


class Ball(pygame.sprite.Sprite):
    """ This class represents the ball
        It derives from the "Sprite" class in Pygame """

    # Speed in pixels per cycle
    speed = 10.0

    # Floating point representation of where the ball is
    x = float(config.ball_x)
    y = float(config.ball_y)

    # Direction of ball (in degrees)
    direction = 200

    width = config.ball_width
    height = config.ball_height

    # Constructor. Pass in the color of the block, and its x and y position
    def __init__(self):
        # Call the parent class (Sprite) constructor
        super().__init__()

        # Create the image of the ball
        self.image = pygame.Surface([self.width, self.height])

        # Color the ball
        self.image.fill(white)

        # Get a rectangle object that shows where our image is
        self.rect = self.image.get_rect()

        # Get attributes for the height/width of the screen
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()

    def bounce(self, diff):
        """ This function will bounce the ball
            off a horizontal surface (not a vertical one) """

        self.direction = (180 - self.direction) % 360
        self.direction -= diff

    def update(self):
        """ Update the position of the ball. """
        # Sine and Cosine work in degrees, so we have to convert them
        direction_radians = math.radians(self.direction)

        # Change the position (x and y) according to the speed and direction
        self.x += self.speed * math.sin(direction_radians)
        self.y -= self.speed * math.cos(direction_radians)

        # Move the image to where our x and y are
        self.rect.x = self.x
        self.rect.y = self.y

        # Do we bounce off the top of the screen?
        if self.y <= 0:
            self.bounce(0)
            self.y = 1

        # Do we bounce off the left of the screen?
        if self.x <= 0:
            self.direction = (360 - self.direction) % 360
            self.x = 1

        # Do we bounce of the right side of the screen?
        if self.x > self.screenwidth - self.width:
            self.direction = (360 - self.direction) % 360
            self.x = self.screenwidth - self.width - 1

        # Did we fall off the bottom edge of the screen?
        if self.y > config.width:
            return True
        else:
            return False


class Player(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the
    player controls. """

    def __init__(self):
        """ Constructor for Player. """
        # Call the parent's constructor
        super().__init__()

        self.width = config.player_width
        self.height = config.player_height
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill((white))

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()

        self.rect.x = config.player_x()
        self.rect.y = self.screenheight - self.height

    def update2(self, dummy):
        """ Update the player position. """
        # Get where the mouse is
        pos = pygame.mouse.get_pos()
        # Set the left side of the player bar to the mouse position
        self.rect.x = pos[0]
        # Make sure we don't push the player paddle
        # off the right side of the screen
        if self.rect.x > self.screenwidth - self.width:
            self.rect.x = self.screenwidth - self.width

    def update(self, action):
        """ Update the player position. """
        action = config.action_scale if action == 0 else -config.action_scale
        self.rect.x += action
        # Make sure we don't push the player paddle
        # off the right side of the screen
        if self.rect.x > self.screenwidth - self.width:
            self.rect.x = self.screenwidth - self.width


class Game:
    def __init__(self):
        # Call this function so the Pygame library can initialize itself
        pygame.init()

        # Create an 800x600 sized screen
        self.screen = pygame.display.set_mode([config.height, config.width])

        # Set the title of the window
        pygame.display.set_caption('Breakout')

        # Enable this to make the mouse disappear when over our window
        pygame.mouse.set_visible(0)

        # This is a font we use to draw text on the screen (size 36)
        self.font = pygame.font.Font(None, config.font_size)

        # Create a surface we can draw on
        self.background = pygame.Surface(self.screen.get_size())

        # Create sprite lists
        self.blocks = pygame.sprite.Group()
        self.balls = pygame.sprite.Group()
        self.allsprites = pygame.sprite.Group()

        # Create the player paddle object
        self.player = Player()
        self.allsprites.add(self.player)

        # Create the ball
        self.ball = Ball()
        self.allsprites.add(self.ball)
        self.balls.add(self.ball)

        # The top of the block (y position)
        top = config.block_start_from_top

        # Number of blocks to create
        blockcount = config.n_block_columns

        # --- Create blocks

        # Five rows of blocks
        for row in range(config.n_block_rows):
            # 32 columns of blocks
            for column in range(0, blockcount):
                # Create a block (color,x,y)
                block = Block(blue, column * (config.block_width + config.block_padding) + 1, top)
                self.blocks.add(block)
                self.allsprites.add(block)
            # Move the top of the next row down
            top += config.block_height + config.block_padding

        # Clock to limit speed
        self.clock = pygame.time.Clock()

        # Is the game over?
        self.game_over = False

        # Exit the program?
        self.exit_program = False

    def get_pixels(self):
        # pixels = np.array(pygame.PixelArray(pygame.display.get_surface()))
        pixels = np.array(pygame.PixelArray(self.screen))
        return pixels

    def display(self):
        pass

        # Draw Everything
        # self.allsprites.draw(self.screen)

        # Flip the screen and show what we've drawn
        # pygame.display.flip()

    def step(self, action):

        # Limit to 30 fps
        self.clock.tick(config.fps)

        # Clear the screen
        self.screen.fill(black)

        # Process the events in the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit_program = True

        # Update the ball and player position as long
        # as the game is not over.
        if not self.game_over:
            # Update the player and ball positions
            self.player.update(action)
            self.game_over = self.ball.update()

        # If we are done, print game over
        if self.game_over:
            text = self.font.render("Game Over", True, white)
            textpos = text.get_rect(centerx=self.background.get_width() / 2)
            textpos.top = config.text_start_from_top
            self.screen.blit(text, textpos)

        # See if the ball hits the player paddle
        if pygame.sprite.spritecollide(self.player, self.balls, False):
            # The 'diff' lets you try to bounce the ball left or right
            # depending where on the paddle you hit it
            diff = (self.player.rect.x + self.player.width / 2) - (self.ball.rect.x + self.ball.width / 2)

            # Set the ball's y position in case
            # we hit the ball on the edge of the paddle
            self.ball.rect.y = self.screen.get_height() - self.player.rect.height - self.ball.rect.height - 1
            self.ball.bounce(diff)

        # Check for collisions between the ball and the blocks
        deadblocks = pygame.sprite.spritecollide(self.ball, self.blocks, True)

        # If we actually hit a block, bounce the ball
        if len(deadblocks) > 0:
            self.ball.bounce(0)

            # Game ends if all the blocks are gone
            if len(self.blocks) == 0:
                self.game_over = True

        # Draw Everything
        self.allsprites.draw(self.screen)

        # Flip the screen and show what we've drawn
        pygame.display.flip()

    def main(self):
        # Main program loop
        while not self.exit_program:
            self.step(None)

        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.main()
