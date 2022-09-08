import random
import pygame
import utils

class SnakeEnv:
    def __init__(self, snake_head_x1, snake_head_y1, snake_head_x2, snake_head_y2, food_x, food_y):
        self.game = Snake(snake_head_x1, snake_head_y1, snake_head_x2, snake_head_y2, food_x, food_y)
        self.render = False

    def get_actions(self):
        return self.game.get_actions()

    def reset(self):
        return self.game.reset()
    
    def get_points(self):
        return self.game.get_points()

    def get_state(self):
        return self.game.get_state()

    def step(self, action1, action2):
        state, points, dead = self.game.step(action1, action2)
        if self.render:
            self.draw(state, points, dead)
        return state, points, dead

    def draw(self, state, points, dead):
        snake_head_x1, snake_head_y1, snake_head_x2, snake_head_y2, snake_body1, snake_body2, food_x, food_y = state
        self.display.fill(utils.BLUE)    
        pygame.draw.rect( self.display, utils.BLACK,
                [
                    utils.GRID_SIZE,
                    utils.GRID_SIZE,
                    utils.DISPLAY_SIZE - utils.GRID_SIZE * 2,
                    utils.DISPLAY_SIZE - utils.GRID_SIZE * 2
                ])

        # draw snake head
        pygame.draw.rect(
                    self.display, 
                    utils.GREEN,
                    [
                        snake_head_x1,
                        snake_head_y1,
                        utils.GRID_SIZE,
                        utils.GRID_SIZE
                    ],
                    3
                )
        pygame.draw.rect(
                    self.display, 
                    utils.YELLOW,
                    [
                        snake_head_x2,
                        snake_head_y2,
                        utils.GRID_SIZE,
                        utils.GRID_SIZE
                    ],
                    3
                )

        # draw snake body
        for seg in snake_body1:
            pygame.draw.rect(
                self.display, 
                utils.GREEN,
                [
                    seg[0],
                    seg[1],
                    utils.GRID_SIZE,
                    utils.GRID_SIZE,
                ],
                1
            )
        for seg in snake_body2:
            pygame.draw.rect(
                self.display, 
                utils.YELLOW,
                [
                    seg[0],
                    seg[1],
                    utils.GRID_SIZE,
                    utils.GRID_SIZE,
                ],
                1
            )

        # draw food
        pygame.draw.rect(
                    self.display, 
                    utils.RED,
                    [
                        food_x,
                        food_y,
                        utils.GRID_SIZE,
                        utils.GRID_SIZE
                    ]
                )

        text_surface = self.font.render("Points: " + str(points), True, utils.BLACK)
        text_rect = text_surface.get_rect()
        text_rect.center = ((280),(25))
        self.display.blit(text_surface, text_rect)
        pygame.display.flip()
        if dead:
            # slow clock if dead
            self.clock.tick(1)
        else:
            self.clock.tick(5)

        return 


    def display(self):
        pygame.init()
        pygame.display.set_caption('MP4: Snake')
        self.clock = pygame.time.Clock()
        pygame.font.init()

        self.font = pygame.font.Font(pygame.font.get_default_font(), 15)
        self.display = pygame.display.set_mode((utils.DISPLAY_SIZE, utils.DISPLAY_SIZE), pygame.HWSURFACE)
        self.draw(self.game.get_state(), self.game.get_points(), False)
        self.render = True
            
class Snake:
    def __init__(self, snake_head_x1, snake_head_y1, snake_head_x2, snake_head_y2, food_x, food_y):
        self.init_snake_head_x1 = snake_head_x1
        self.init_snake_head_y1 = snake_head_y1
        self.init_snake_head_x2 = snake_head_x2
        self.init_snake_head_y2 = snake_head_y2
        self.init_food_x = food_x
        self.init_food_y = food_y
        self.reset()

    def reset(self):
        self.points1 = 0
        self.points2 = 0
        self.snake_head_x1 = self.init_snake_head_x1
        self.snake_head_y1 = self.init_snake_head_y1
        self.snake_head_x2 = self.init_snake_head_x2
        self.snake_head_y2 = self.init_snake_head_y2
        self.snake_body1 = []
        self.snake_body2 = []
        self.food_x = self.init_food_x
        self.food_y = self.init_food_y

    def get_points(self):
        return self.points1

    def get_actions(self):
        return [0, 1, 2, 3]

    def get_state(self):
        return [
            self.snake_head_x1,
            self.snake_head_y1,
            self.snake_head_x2,
            self.snake_head_y2,
            self.snake_body1,
            self.snake_body2,
            self.food_x,
            self.food_y
        ]

    def move(self, action1, action2):
        delta_x1 = delta_y1 = 0
        delta_x2 = delta_y2 = 0
        if action1 == 0:
            delta_y1 = -1 * utils.GRID_SIZE
        elif action1 == 1:
            delta_y1 = utils.GRID_SIZE
        elif action1 == 2:
            delta_x1 = -1 * utils.GRID_SIZE
        elif action1 == 3:
            delta_x1 = utils.GRID_SIZE
        
        if action2 == 0:
            delta_y2 = -1 * utils.GRID_SIZE
        elif action2 == 1:
            delta_y2 = utils.GRID_SIZE
        elif action2 == 2:
            delta_x2 = -1 * utils.GRID_SIZE
        elif action2 == 3:
            delta_x2 = utils.GRID_SIZE

        old_body_head1 = None
        if len(self.snake_body1) == 1:
            old_body_head1 = self.snake_body1[0]
        self.snake_body1.append((self.snake_head_x1, self.snake_head_y1))
        self.snake_head_x1 += delta_x1
        self.snake_head_y1 += delta_y1

        old_body_head2 = None
        if len(self.snake_body2) == 1:
            old_body_head2 = self.snake_body2[0]
        self.snake_body2.append((self.snake_head_x2, self.snake_head_y2))
        self.snake_head_x2 += delta_x2
        self.snake_head_y2 += delta_y2

        # if len(self.snake_body) > self.points:
        #     del(self.snake_body[0])

        self.handle_eatfood()

        # colliding with the snake body or going backwards while its body length
        # greater than 1
        if len(self.snake_body1) >= 1:
            for seg in self.snake_body1:
                if self.snake_head_x1 == seg[0] and self.snake_head_y1 == seg[1]:
                    return True
                if self.snake_head_x2 == seg[0] and self.snake_head_y2 == seg[1]:
                    return True
        if len(self.snake_body2) >= 1:
            for seg in self.snake_body2:
                if self.snake_head_x2 == seg[0] and self.snake_head_y2 == seg[1]:
                    return True
                if self.snake_head_x1 == seg[0] and self.snake_head_y1 == seg[1]:
                    return True

        # moving towards body direction, not allowing snake to go backwards while 
        # its body length is 1
        if len(self.snake_body1) == 1:
            if old_body_head1 == (self.snake_head_x1, self.snake_head_y1):
                return True
        if len(self.snake_body2) == 1:
            if old_body_head2 == (self.snake_head_x2, self.snake_head_y2):
                return True

        # collide with the wall
        if (self.snake_head_x1 < utils.GRID_SIZE or self.snake_head_y1 < utils.GRID_SIZE or
            self.snake_head_x1 + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE or self.snake_head_y1 + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE):
            return True
        if (self.snake_head_x2 < utils.GRID_SIZE or self.snake_head_y2 < utils.GRID_SIZE or
            self.snake_head_x2 + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE or self.snake_head_y2 + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE):
            return True

        return False

    def step(self, action1, action2):
        is_dead = self.move(action1, action2)
        return self.get_state(), self.get_points(), is_dead

    def handle_eatfood(self):
        generate_new_food = False
        if (self.snake_head_x1 == self.food_x) and (self.snake_head_y1 == self.food_y):
            self.points1 += 1
            self.points2 -= 1
            generate_new_food = True
        if (self.snake_head_x2 == self.food_x) and (self.snake_head_y2 == self.food_y):
            self.points1 -= 1
            self.points2 += 1
            generate_new_food = True
        
        if generate_new_food:
            self.random_food()

    def random_food(self):
        max_x = (utils.DISPLAY_SIZE - utils.WALL_SIZE - utils.GRID_SIZE)
        max_y = (utils.DISPLAY_SIZE - utils.WALL_SIZE - utils.GRID_SIZE)
        
        self.food_x = random.randint(utils.WALL_SIZE, max_x)//utils.GRID_SIZE * utils.GRID_SIZE
        self.food_y = random.randint(utils.WALL_SIZE, max_y)//utils.GRID_SIZE * utils.GRID_SIZE

        while self.check_food_on_snake():
            self.food_x = random.randint(utils.WALL_SIZE, max_x)//utils.GRID_SIZE * utils.GRID_SIZE
            self.food_y = random.randint(utils.WALL_SIZE, max_y)//utils.GRID_SIZE * utils.GRID_SIZE

    def check_food_on_snake(self):
        if self.food_x == self.snake_head_x1 and self.food_y == self.snake_head_y1:
            return True 
        for seg in self.snake_body1:
            if self.food_x == seg[0] and self.food_y == seg[1]:
                return True

        if self.food_x == self.snake_head_x2 and self.food_y == self.snake_head_y2:
            return True 
        for seg in self.snake_body2:
            if self.food_x == seg[0] and self.food_y == seg[1]:
                return True
        return False
        
    
