import pygame
import numpy as np
import enum

from DQN.catcher.env_config import Config1 as config

# Define some colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)

# Size of break-out blocks
block_width = config.block_width
block_height = config.block_height


class Ball(pygame.sprite.Sprite):
    """ This class represents the ball
        It derives from the "Sprite" class in Pygame """

    # Speed in pixels per cycle
    speed = 3

    # Floating point representation of where the ball is
    x = float(config.ball_x)
    y = float(config.ball_y)

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
        self.reset()

    def update(self):
        """ Update the position of the ball. """

        # Change the position (x and y) according to the speed and direction
        self.y += self.speed

        # Move the image to where our x and y are
        self.rect.x = self.x
        self.rect.y = self.y

        # Did we fall off the bottom edge of the screen?
        if self.y > config.height:
            return True
        else:
            return False

    def reset(self):
        self.y = 0
        self.x = np.random.randint(1, self.screenwidth - self.width)


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

        self.rect.x = self.screenwidth / 2 - self.width / 2
        self.rect.y = self.screenheight - self.height
        self.n_lives = 5

    def update(self, action):
        """ Update the player position. """
        action = config.action_scale if action == 0 else -config.action_scale
        self.rect.x += action
        # Make sure we don't push the player paddle
        # off the right side of the screen
        if self.rect.x > self.screenwidth - self.width:
            self.rect.x = self.screenwidth - self.width


class Game:
    class StepRes(enum.Enum):
        CAUGHT = 0
        DROPPED = 1
        NA = 2

    def __init__(self):
        # Call this function so the Pygame library can initialize itself
        pygame.init()

        # Create an 800x600 sized screen
        self.screen = pygame.display.set_mode([config.height, config.width])

        # Set the title of the window
        pygame.display.set_caption('Catcher')

        # Enable this to make the mouse disappear when over our window
        pygame.mouse.set_visible(0)

        # This is a font we use to draw text on the screen (size 36)
        self.font = pygame.font.Font(None, config.font_size)

        # Create a surface we can draw on
        self.background = pygame.Surface(self.screen.get_size())

        # Create sprite lists
        self.balls = pygame.sprite.Group()
        self.allsprites = pygame.sprite.Group()

        # Create the player paddle object
        self.player = Player()
        self.allsprites.add(self.player)

        # Create the ball
        self.ball = Ball()
        self.allsprites.add(self.ball)
        self.balls.add(self.ball)

        # Clock to limit speed
        self.clock = pygame.time.Clock()

        # Is the game over?
        self.game_over = False

        # Exit the program?
        self.exit_program = False

    def get_pixels(self):
        pixels = np.array(pygame.PixelArray(self.screen))
        return pixels

    def display(self):
        # Draw Everything
        self.allsprites.draw(self.screen)

        # Flip the screen and show what we've drawn
        pygame.display.flip()

    def step(self, action):
        result = self.StepRes.NA

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
            lost = self.ball.update()
            self.player.n_lives -= lost
            self.game_over = self.player.n_lives <= 0

            if not self.game_over and lost:
                self.ball.reset()
                result = self.StepRes.DROPPED

        # If we are done, print game over
        if self.game_over:
            text = self.font.render("Game Over", True, white)
            textpos = text.get_rect(centerx=self.background.get_width() / 2)
            textpos.top = config.text_start_from_top
            self.screen.blit(text, textpos)

        # See if the ball hits the player paddle
        if pygame.sprite.spritecollide(self.player, self.balls, False):
            self.ball.reset()
            result = self.StepRes.CAUGHT

        # Check for collisions between the ball and the blocks
        # Draw Everything
        self.allsprites.draw(self.screen)

        # Flip the screen and show what we've drawn
        pygame.display.flip()

        return result

    def main(self):
        # Main program loop
        while not self.exit_program:
            self.step(np.random.randint(2))

        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.main()
