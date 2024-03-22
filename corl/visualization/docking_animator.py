import numpy as np
from plotly.graph_objs import Scatter, Scatter3d

from corl.dones.docking_1d.dones import DockingDoneFunction
from corl.visualization.plotly_animator import PlotlyTrajectoryAnimation, draw_custom_polygon


class Docking1dAnimation(PlotlyTrajectoryAnimation):
    def _draw_custom_polygons(self, _) -> list[Scatter | Scatter3d] | None:
        """
        Adds the docking region and maximum velocity threshold for successful docking to the trajectory animation
        """
        agents = self.config.episode_artifact.env_config["agents"]
        custom_polygons = []
        for agent_id, agent_name in self.config.episode_artifact.agent_to_platforms.items():
            agent_config = agents[agent_id].class_config.config
            done_config = [(done["config"]) for done in agent_config["dones"] if done["name"] == DockingDoneFunction.__name__]
            docking_region_radius = done_config[0]["docking_region_radius"]["value"]
            done_config[0]["docking_region_radius"]["units"]

            velocity_threshold = done_config[0]["velocity_threshold"]["value"]
            velocity_units = done_config[0]["velocity_threshold"]["units"]

            x_coordinates = []
            y_coordinates = []
            t = 0.0
            while t <= 2 * np.pi:
                x_coordinates.append(docking_region_radius * np.cos(t))
                y_coordinates.append(docking_region_radius * np.sin(t))
                t += 0.1
            custom_polygons.append(
                draw_custom_polygon(
                    x_coordinates=x_coordinates,
                    y_coordinates=y_coordinates,
                    fill=False,
                    color=self.config.platform_color_map[agent_name[0]] if self.config.platform_color_map else "black",
                    name=f"Docking Region - velocity threshold: {velocity_threshold} [{velocity_units}]",
                )
            )
        return custom_polygons
