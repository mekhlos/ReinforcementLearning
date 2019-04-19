import pygame
import pygame.gfxdraw


class Agent(pygame.sprite.Sprite):
    def __init__(self, position, velocity, r, behaviour):
        super().__init__()

        self.image = pygame.Surface([2 * r + 2, 2 * r + 2])
        pygame.gfxdraw.filled_circle(self.image, r, r, r, (255, 255, 255))
        self.rect = self.image.get_rect()

        self.position = position
        self.velocity = velocity
        self.r = r

        self.rect.x = position[0]
        self.rect.y = position[1]

        self.behaviour = behaviour

    def update(self, *args):
        screen_height = pygame.display.get_surface().get_height()
        screen_width = pygame.display.get_surface().get_width()

        new_x, new_y = self.behaviour.get_new_position((self.rect.x, self.rect.y))

        self.rect.x = new_x % screen_width
        self.rect.y = new_y % screen_height
