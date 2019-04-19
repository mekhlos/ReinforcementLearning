import pygame

# Colors
BLACK = (0, 0, 0)


class World:
    def __init__(self, width, height, agents):
        self.agents = agents
        self.height = height
        self.width = width

        pygame.init()

        # Create a height x width sized screen
        self.screen = pygame.display.set_mode([height, width])

        # Set the title of the window
        pygame.display.set_caption('My world')

        # Create a surface we can draw on
        self.background = pygame.Surface(self.screen.get_size())

        # Create sprite lists
        self.agent_sprites = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()

        self.agent_sprites.add(agents)
        self.all_sprites.add(self.agent_sprites)

        # Clock to limit speed
        self.clock = pygame.time.Clock()

    def run(self):
        should_exit = False

        while not should_exit:
            # Limit to 30 fps
            self.clock.tick(30)

            # Clear the screen
            self.screen.fill(BLACK)

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_exit = True

            self.agent_sprites.update()

            # Draw Everything
            self.all_sprites.draw(self.screen)

            # Flip the screen and show what we've drawn
            pygame.display.flip()

        pygame.quit()
