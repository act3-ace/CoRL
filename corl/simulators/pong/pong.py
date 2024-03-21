"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Pong Game
This class implements pong and is independing of RL based on
https://github.com/techwithtim/Pong-Python
"""
from enum import Enum

import pygame
from pydantic import BaseModel, ConfigDict


class Colors(Enum):
    """
    Enumer ations of display colors
    """

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)


class GameStatus(Enum):
    """
    Enumeration of game status
    """

    LEFT_WIN = 0
    RIGHT_WIN = 1
    IN_PROGRESS = 2


class Paddle(BaseModel):
    """
    paddle controlled by user
    """

    width: int = 20
    height: int = 100
    x: float = 0
    y: float = 0
    vel: float = 4
    color: tuple[int, int, int] = Colors.WHITE.value
    collision_on: bool = True
    health: int = 1
    current_health: int = 1
    ball_hits: int = 0

    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.reset()
        self.current_health = self.health

    def move(self, up=True):
        """
        move paddle y position up or down by velocity value
        """
        if up:
            self.y -= self.vel
        else:
            self.y += self.vel

    def reset(self):
        """
        move paddle to initial position
        """
        self.current_health = self.health
        self.collision_on = True


class Ball(BaseModel):
    """
    Ball object
    """

    max_vel: int = 5
    radius: int = 7
    x: float = 0
    y: float = 0
    x_vel: float = 5
    y_vel: float = 0

    def move(self):
        """
        move ball using current velocity
        """
        self.x += self.x_vel
        self.y += self.y_vel


class PongRender:
    """
    Class to render the current pong game
    """

    def __init__(self, screen_width, screen_height):
        self.display_window = pygame.display.set_mode((screen_width, screen_height))
        self.score_font = pygame.font.SysFont("comicsans", 50)
        self.extra_info_font = pygame.font.SysFont("comicsans", 24)
        self.ball_color = Colors.WHITE.value
        self.left_paddle_color = Colors.BLUE.value
        self.right_paddle_color = Colors.RED.value

    def draw(self, pong, left_score, right_score, extra_info=None):
        """
        draw all objects and scores on display
        """

        self.display_window.fill(Colors.BLACK.value)
        left_score_text = self.score_font.render(f"{left_score}", 1, Colors.WHITE.value)
        right_score_text = self.score_font.render(f"{right_score}", 1, Colors.WHITE.value)
        self.display_window.blit(left_score_text, (self.display_window.get_width() // 4 - left_score_text.get_width() // 2, 20))
        self.display_window.blit(right_score_text, (self.display_window.get_width() * (3 / 4) - right_score_text.get_width() // 2, 20))

        if extra_info:
            up_down_offset = self.display_window.get_height() * 0.95
            for item in extra_info:
                extra_info_text = self.extra_info_font.render(item, 1, Colors.WHITE.value)
                text_left_right_pos = max(self.display_window.get_width() * 0.1 - extra_info_text.get_width() // 2, 0.05)
                text_up_down_pos = up_down_offset - extra_info_text.get_height()
                self.display_window.blit(extra_info_text, (text_left_right_pos, text_up_down_pos))
                up_down_offset = text_up_down_pos

        pygame.draw.rect(
            self.display_window,
            self.left_paddle_color,
            (pong.left_paddle.x, pong.left_paddle.y, pong.left_paddle.width, pong.left_paddle.height),
        )
        pygame.draw.rect(
            self.display_window,
            self.right_paddle_color,
            (pong.right_paddle.x, pong.right_paddle.y, pong.right_paddle.width, pong.right_paddle.height),
        )

        for i in range(10, self.display_window.get_height(), self.display_window.get_height() // 20):
            if i % 2 == 1:
                continue
            pygame.draw.rect(
                self.display_window,
                Colors.WHITE.value,
                (self.display_window.get_width() // 2 - 5, i, 10, self.display_window.get_height() // 20),
            )

        pygame.draw.circle(self.display_window, self.ball_color, (pong.ball.x, pong.ball.y), pong.ball.radius)

        pygame.display.update()

    def draw_win(self, win_text):
        """
        draw win text
        """
        text = self.score_font.render(win_text, 1, Colors.WHITE.value)
        self.display_window.blit(
            text,
            (self.display_window.get_width() // 2 - text.get_width() // 2, self.display_window.get_height() // 2 - text.get_height() // 2),
        )

        pygame.display.update()
        pygame.time.delay(5000)


class Pong(BaseModel):
    """
    Pong game
    """

    screen_width: int = 700
    screen_height: int = 500
    left_paddle: Paddle = Paddle()
    right_paddle: Paddle = Paddle()
    ball: Ball = Ball()
    health_enabled: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.reset()

    def reset(self):
        """
        reset game
        """
        self.left_paddle.x = 10
        self.left_paddle.y = self.screen_height // 2 - self.left_paddle.height // 2
        self.right_paddle.x = self.screen_width - 10 - self.right_paddle.width
        self.right_paddle.y = self.screen_height // 2 - self.right_paddle.height // 2
        self.ball.x = self.screen_width // 2
        self.ball.y = self.screen_height // 2

    def handle_paddle_movement(self, keys, left_paddle, right_paddle):
        """Move paddle positions based on keyboard values

        Parameters
        ----------
        keys : list
            list of pygame key presses
        left_paddle : paddle
            left paddle in game
        right_paddle : Ball
            right pad  in game
        """

        if pygame.K_w in keys and left_paddle.y - left_paddle.vel >= 0:
            left_paddle.move(up=True)
        if pygame.K_s in keys and left_paddle.y + left_paddle.vel + left_paddle.height <= self.screen_height:
            left_paddle.move(up=False)

        if pygame.K_UP in keys and right_paddle.y - right_paddle.vel >= 0:
            right_paddle.move(up=True)
        if pygame.K_DOWN in keys and right_paddle.y + right_paddle.vel + right_paddle.height <= self.screen_height:
            right_paddle.move(up=False)

    def handle_collision(self, ball, left_paddle, right_paddle):
        """Update ball velocity based on ball and paddle positions

        Parameters
        ----------
        ball : Ball
            ball in pong game
        left_paddle : paddle
            left paddle in game
        right_paddle : Ball
            right pad  in game
        """

        if ball.y + ball.radius >= self.screen_height or ball.y - ball.radius <= 0:
            ball.y_vel *= -1

        if ball.x_vel < 0:
            if (
                left_paddle.collision_on
                and ball.y >= left_paddle.y
                and ball.y <= left_paddle.y + left_paddle.height
                and ball.x - ball.radius <= left_paddle.x + left_paddle.width
            ):
                ball.x_vel *= -1

                middle_y = left_paddle.y + left_paddle.height / 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (left_paddle.height / 2) / ball.max_vel
                y_vel = difference_in_y / reduction_factor
                ball.y_vel = -1 * y_vel
                if self.health_enabled:
                    left_paddle.current_health -= 1
                    if left_paddle.current_health <= 0:
                        left_paddle.collision_on = False
                # Updates the ball hit count for the left paddle
                left_paddle.ball_hits += 1

        elif (
            right_paddle.collision_on
            and ball.y >= right_paddle.y
            and ball.y <= right_paddle.y + right_paddle.height
            and ball.x + ball.radius >= right_paddle.x
        ):
            ball.x_vel *= -1

            middle_y = right_paddle.y + right_paddle.height / 2
            difference_in_y = middle_y - ball.y
            reduction_factor = (right_paddle.height / 2) / ball.max_vel
            y_vel = difference_in_y / reduction_factor
            ball.y_vel = -1 * y_vel
            if self.health_enabled:
                right_paddle.current_health -= 1
                if right_paddle.current_health <= 0:
                    right_paddle.collision_on = False
            # Updates the ball hit count for the right paddle
            right_paddle.ball_hits += 1

    def step(self, keys: list) -> GameStatus:
        """Advance pong game

        Parameters
        ----------
        keys : list
            list of keys pressed on keyboard.

        Returns
        -------
        GameStatus
            status of game after step
        """

        game_status = GameStatus.IN_PROGRESS

        self.handle_paddle_movement(keys, self.left_paddle, self.right_paddle)

        self.ball.move()

        self.handle_collision(self.ball, self.left_paddle, self.right_paddle)

        if self.ball.x < 0:
            game_status = GameStatus.RIGHT_WIN
        elif self.ball.x > self.screen_width:
            game_status = GameStatus.LEFT_WIN

        return game_status


def main():
    """
    Interactive game
    """
    pygame.init()

    run = True
    clock = pygame.time.Clock()
    fps = 60

    left_score = 0
    right_score = 0
    winning_score = 1

    pong = Pong(screen_width=700, screen_height=500, health_enabled=True)
    pong_render = PongRender(pong.screen_width, pong.screen_height)

    while run:
        clock.tick(fps)
        pong_render.draw(pong, left_score, right_score)

        keys = pygame.key.get_pressed()

        pressed_keys = [key for key in [pygame.K_w, pygame.K_s, pygame.K_UP, pygame.K_DOWN] if keys[key]]

        game_status = pong.step(pressed_keys)

        if game_status is GameStatus.RIGHT_WIN:
            right_score += 1
            pong.reset()
        elif game_status is GameStatus.LEFT_WIN:
            left_score += 1
            pong.reset()

        won = False
        if left_score >= winning_score:
            won = True
            win_text = "Left Player Won!"
        elif right_score >= winning_score:
            won = True
            win_text = "Right Player Won!"

        if won:
            pong_render.draw_win(win_text)
            pong.reset()
            left_score = 0
            right_score = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

    pygame.quit()


if __name__ == "__main__":
    main()
