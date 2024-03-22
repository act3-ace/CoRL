import numpy as np
from plotly.graph_objs import Scatter, Scatter3d
from pygame import Rect

from corl.simulators.pong.pong import Ball
from corl.visualization.plotly_animator import PlotlyTrajectoryAnimation, draw_custom_polygon


class PongAnimation(PlotlyTrajectoryAnimation):
    def _draw_custom_polygons(self, step_idx) -> list[Scatter | Scatter3d] | None:
        polygons = []
        for platform_serialized_data in self.config.episode_artifact.steps[step_idx].platforms:
            agent_name = platform_serialized_data["name"]
            paddle_height = platform_serialized_data["paddle_height"]
            paddle_width = platform_serialized_data["paddle_width"]
            paddle_position = platform_serialized_data["position"]

            rect = Rect(paddle_position[0], paddle_position[1], paddle_width, paddle_height)
            x_coordinates = [rect.topleft[0], rect.topright[0], rect.bottomright[0], rect.bottomleft[0]]
            y_coordinates = [rect.topleft[1], rect.topright[1], rect.bottomright[1], rect.bottomleft[1]]
            polygon = draw_custom_polygon(
                x_coordinates=x_coordinates,
                y_coordinates=y_coordinates,
                color=self.config.platform_color_map[agent_name] if self.config.platform_color_map else "black",
                name=f"{agent_name} area",
            )
            polygons.append(polygon)

        data = self.config.episode_artifact.steps[step_idx].platforms[0]
        ball_measurement = data["sensors"]["Ball_Sensor"]["measurement"].m
        ball_radius = Ball().radius
        ball_x_coordinates = []
        ball_y_coordinates = []
        ball_x = ball_measurement[0]
        ball_y = ball_measurement[1]
        t = 0.0
        while t <= 2 * np.pi:
            ball_x_coordinates.append(ball_radius * np.cos(t) + ball_x)
            ball_y_coordinates.append(ball_radius * np.sin(t) + ball_y)
            t += 0.1

        ball = draw_custom_polygon(x_coordinates=ball_x_coordinates, y_coordinates=ball_y_coordinates, name="ball")
        polygons.append(ball)

        return polygons
