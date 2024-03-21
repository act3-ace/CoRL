"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Print a visualization to screen
"""
import os
import pathlib
import typing

import pandas as pd

# from dash import Dash, dcc, html, Input, Output
try:
    import plotly.express as px

    html_plots = True
except ImportError as e:
    print("HTMl Plots unavailable: ", e)
    html_plots = False

from corl.evaluation.metrics.types.nonterminals.timed_value import TimedValue
from corl.evaluation.metrics.types.nonterminals.vector import Vector

# from corl.evaluation.visualization.plot_configuration import PlotConfiguration, RangeMetric, RangePlot
from corl.evaluation.visualization.visualization import Visualization
from corl.libraries.units import Quantity

FUTURE = typing.Any


class HMTLPlots(Visualization):
    """Visualizer that prints to stdout"""

    def __init__(
        self,  # plot_configurations: typing.List[PlotConfiguration],
        *args,
        **kwargs,
    ):
        """Setup a Jupyter notebook visualization

        Parameters
        ----------
        done_means: str
            Path to checkpoint file
        done_variances: str
            File name that is being used during this evaluation
        """
        super().__init__(*args, **kwargs)
        # self._plot_configurations = plot_configurations

        if html_plots is False:
            raise RuntimeError("Can not utilize HTMLPlots with no plotly")

    def generate_plot(
        self,
        df,
    ) -> None:
        """Generate html plot using plotly"""

        filename = str(df.columns[0])

        # Figure out the title and directory_path for figure
        if len(df.columns) == 0:
            raise RuntimeError("Zero columns provided")
        if len(df.columns) == 1:
            title = filename
            participant_name, event_data, metric = filename.split("--")
            directory_path = os.path.join(self.out_folder, f"test_case_{event_data!s}", "plots", str(participant_name))
        else:
            all_names: list[str] = list(df.columns)

            # we now have a list of names, how should we combine?
            # We are going to assume the format is participant--test_case--metric
            # going to further assume that participant_name is the same and metric is the same
            participant_name, _, metric = all_names[0].split("--")

            # TODO: is something better here?
            #       Should we do some error handling to make sure everyones participant_name and metric are the same?

            title = str(participant_name + "--" + str(metric))
            directory_path = os.path.join(self.out_folder, "plots", str(participant_name))

        # Generate the figure
        fig = px.line(df, title=title)

        # Write figure to html
        pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
        print("My current directory is : " + directory_path)
        fig.write_html(os.path.join(directory_path, str(metric) + ".html"))

    def __extract_evaluation_metric_as_dataframes(self) -> dict[str, FUTURE]:
        """This function creates a plot for each metric that contains all the runs, for each participant"""
        participant_dict: dict = {}
        metric_dict: dict = {}

        for participant in self._post_processed_data.participants:
            print(str(participant))
            for event in self._post_processed_data.participants[participant].events:
                # Search for metric on each event of the participant
                for metric, value in event.metrics.items():
                    # Can only plot a
                    if isinstance(value, Vector) and len(value.arr) > 0 and isinstance(value.arr[0], TimedValue):
                        arr: list[TimedValue] = value.arr  # type: ignore

                        if isinstance((arr[0].value.value), Quantity):  # type: ignore
                            data = list(zip([x.time.value for x in arr], [x.value.value.m for x in arr]))  # type: ignore
                        elif isinstance((value.arr[0].value.value[0]), float):  # type: ignore
                            data = list(zip([x.time.value for x in arr], [x.value.value[0] for x in arr]))  # type: ignore
                        else:
                            raise RuntimeError("Unknown how to extract")
                            # data = list(zip([x.time.value for x in value.arr], [x.value.value[0] for x in value.arr]))
                            # data = list(zip([x.time.value for x in value.arr], [x.value.value.value for x in value.arr]))

                        label = str(participant) + "--" + str(event.data.test_case) + "--" + str(metric)

                        df = pd.DataFrame(data, columns=["time", label])
                        df.set_index("time", inplace=True)
                        if metric not in metric_dict:
                            print("New metric found: " + str(metric))
                            metric_dict[metric] = []
                            metric_dict[metric].append(df)
                        else:
                            print("Metric " + str(metric) + " already exists. Append")
                            metric_dict[metric].append(df)

            # copy the dict so you won't lose data once its cleared
            complete_dict = metric_dict.copy()
            # clear metric_dict so it will be empty for the next participant
            metric_dict.clear()

            if str(participant) not in participant_dict:
                participant_dict[participant] = complete_dict
            else:
                participant_dict[participant].append(complete_dict)

        return participant_dict

    def visualize(self):  # I am okay with this method being long
        """Execute visualization"""

        # plot_list = self.__extract_event_metric_as_dataframes()
        # for range_df in plot_list:
        #     self.generate_plot(range_df, Filename.EVENT)

        # Generate dataframes for the plots we are going to generate
        plot_dict = self.__extract_evaluation_metric_as_dataframes()

        for plot_data in plot_dict.values():
            for metric in plot_data:
                # create aggregate dataframe to aggregate across test cases
                aggregate_df = pd.DataFrame(columns=["time"])
                aggregate_df.set_index("time", inplace=True)

                # Iterate over test cases
                for item in plot_data[metric]:
                    # generate individual test case plot
                    self.generate_plot(item)

                    # Add the data to the aggregate
                    aggregate_df = pd.concat([aggregate_df, item])

                # Generate the aggregate plot
                self.generate_plot(aggregate_df)
