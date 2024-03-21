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
import io
import os

from tabulate import tabulate

from corl.evaluation import metrics
from corl.evaluation.metrics.alerts import Alert
from corl.evaluation.metrics.types.nonterminals.dict import Dict
from corl.evaluation.metrics.types.nonterminals.timed_value import TimedValue
from corl.evaluation.metrics.types.nonterminals.vector import Vector
from corl.evaluation.metrics.types.terminals.discrete import Discrete
from corl.evaluation.metrics.types.terminals.rate import Rate
from corl.evaluation.metrics.types.terminals.real import Real
from corl.evaluation.metrics.types.terminals.string import String
from corl.evaluation.metrics.types.terminals.void import Void
from corl.evaluation.visualization.visualization import Visualization


class Print(Visualization):
    """Visualizer that prints to stdout"""

    def __init__(
        self,
        event_table_print: bool = False,
        event_table_field_skip: list[str] | None = None,
        *args,
        **kwargs,
    ):
        """Print data as a event table to both terminal and file"""
        super().__init__(*args, **kwargs)

        self.event_table_print = event_table_print

        if event_table_field_skip is None:
            self.event_table_field_skip = []
        else:
            self.event_table_field_skip = event_table_field_skip
        # self.event_table_field_skip = []

    def metric_prints(
        self, name: str, metric: metrics.metric.Metric, multiple_lines: bool = False, indent: str = "  ", indent_level: int = 0
    ) -> str:
        """Formats a metric type to console print"""
        if isinstance(metric, Discrete):
            string = str(metric.value.name)
        elif isinstance(metric, Real):
            string = str(metric.value)
        elif isinstance(metric, String):
            string = metric.value
        elif isinstance(metric, Rate):
            string = f"{metric.occurrence}/{metric.total}"
        elif isinstance(metric, Vector):
            new_line = "\n"
            string = f"{new_line.join([self.metric_prints(name, item) for item in metric.arr])}"
            if len(metric.arr) > 10:
                with open(os.path.join(self.out_folder, f"{name}.csv"), "w", encoding="utf-8") as fp:
                    fp.write(string.replace(" - ", ","))
                string = f"<{fp.name}>"
            else:
                string = string.replace(",", ",\n")
        elif isinstance(metric, TimedValue):
            string = f"{metric.time} - {self.metric_prints(name, metric.value)}"
        elif isinstance(metric, Dict):
            if multiple_lines:
                internal_data = {
                    key: self.metric_prints(
                        name=name, metric=value, multiple_lines=multiple_lines, indent=indent, indent_level=indent_level + 1
                    )
                    for key, value in metric.data.items()
                }
                string = "".join([f"\n{indent*(indent_level+1)}{key}: {value}" for key, value in internal_data.items()])
            else:
                string = f"{metric.data}"
        elif isinstance(metric, Void):
            string = "No Data"
        else:
            raise RuntimeError(f"Unknown metric type {type(metric)}")

        return string

    # I am okay with this method being long
    def visualize(self) -> None:  # noqa: PLR0915
        """Execute visualization"""

        # ###########################
        # Generate Summary

        summary_buf = io.StringIO()

        if len(self._post_processed_data.world.alerts) > 0:
            print("#################################", file=summary_buf)
            print("Alerts", file=summary_buf)
            alert_types: dict[str, list[Alert]] = {}
            for alert in self._post_processed_data.world.alerts:
                if alert.type not in alert_types:
                    alert_types[alert.type] = []
                alert_types[alert.type].append(alert)

            for alert_type, alert_items in alert_types.items():
                print("  ", alert_type, file=summary_buf)
                for item in alert_items:
                    print("    ", item.name, file=summary_buf)

        print("#################################", file=summary_buf)
        print("Metrics:", file=summary_buf)

        # print any meta metrics
        # print("  meta:")
        for evaluation_metric in self._post_processed_data.world.metrics:
            if evaluation_metric in self.event_table_field_skip:
                continue
            print(
                f"    {evaluation_metric}: "
                f"{self.metric_prints(evaluation_metric, self._post_processed_data.world.metrics[evaluation_metric])}",
                file=summary_buf,
            )

        # Iterate over each agent and print
        for policy_id in self._post_processed_data.participants:
            print(f"  {policy_id}:", file=summary_buf)

            for evaluation_metric in self._post_processed_data.participants[policy_id].metrics:
                if evaluation_metric in self.event_table_field_skip:
                    continue
                printed_data = self.metric_prints(
                    evaluation_metric,
                    self._post_processed_data.participants[policy_id].metrics[evaluation_metric],
                    multiple_lines=True,
                    indent="  ",
                    indent_level=2,
                )
                print(f"    {evaluation_metric}: {printed_data}", file=summary_buf)

        print(summary_buf.getvalue())

        evaluation_summary_file = self.out_folder / "evaluation_summary.txt"
        with open(evaluation_summary_file, "w", encoding="utf-8") as fp:
            fp.write(summary_buf.getvalue())

            print(f"Evaluation Summary written to {evaluation_summary_file}")

        # ###########################
        # Generate Event Table

        # Print Event Table
        headers = ["meta"]
        headers.extend(list(self._post_processed_data.participants))

        table_data = []
        for idx, event in enumerate(self._post_processed_data.world.events):
            row_data = []

            sub_table_headers = ["metric", "value"]
            sub_table_data = []
            for metric in self._post_processed_data.world.events[idx].metrics:
                if metric in self.event_table_field_skip:
                    continue
                sub_table_data.append(
                    [metric, self.metric_prints(metric, self._post_processed_data.world.events[idx].metrics[metric], True)]
                )

            row_data.append(tabulate(sub_table_data, sub_table_headers))

            # Put each agents metrics in the table
            for agent in self._post_processed_data.participants:
                cell = []
                sub_table_headers = ["metric", "value"]
                sub_table_data = []
                for metric_name in self._post_processed_data.participants[agent].events[idx].metrics:
                    if metric_name in self.event_table_field_skip:
                        continue

                    metric = self._post_processed_data.participants[agent].events[idx].metrics[metric_name]  # type: ignore

                    if isinstance(metric, metrics.types.nonterminals.vector.Vector):
                        if len(metric.arr) > 10:
                            # TODO clong: this code block is unfixable and needs to be rewritten
                            #         # If everything is timed value then save to CSV
                            #         if all([isinstance(item, metrics.types.nonterminals.timed_value.TimedValue) for item in metric.arr]):

                            #             csv_file_name = self.out_folder / f"test_case_{idx}/{metric_name}.csv"
                            #             with open(csv_file_name, "w", encoding="utf-8") as fp:

                            #                 # check that make sure all units are right

                            #                 # Write units as header to csv
                            #                 assert isinstance(metric.arr[0], metrics.types.nonterminals.timed_value.TimedValue)
                            #                 fp.write(f"{metric.arr[0].time.u[1][0]},{metric.arr[0].value.value.units.value[1][0]}\n")

                            #                 # Write line by line with no units
                            #                 for item in metric.arr:
                            #                     fp.write(f"{item.time.value},{item.value.value.value}\n")
                            #             sub_table_data.append([metric_name, f"<{csv_file_name}>"])
                            #         else:
                            sub_table_data.append([metric_name, "<skipped>"])
                        continue

                    sub_table_data.append(
                        [
                            metric_name,
                            self.metric_prints(
                                metric_name, self._post_processed_data.participants[agent].events[idx].metrics[metric_name], True
                            ),
                        ]
                    )

                    cell.append(
                        f"{metric_name}: {self.metric_prints(metric_name, self._post_processed_data.participants[agent].events[idx].metrics[metric_name], True)}"  # noqa
                    )

                row_data.append(tabulate(sub_table_data, sub_table_headers))

            # Now the header for each entry in the table will be by test_case_number
            table_data.extend([[f"Test_Case_{event._data.test_case}"], row_data])  # noqa: SLF001

            table_str = tabulate(table_data, headers, "fancy_grid")
            table_str = table_str.replace("\n", "\n    ")

        if self.event_table_print:
            print(f"    {table_str}")

        if not self.out_folder.exists():
            self.out_folder.mkdir()

        event_table_file = self.out_folder / "event_table.txt"
        with open(event_table_file, "w", encoding="utf-8") as fp:
            fp.write("    " + table_str)

            print(f"Evaluation Event Table written to {event_table_file}")
