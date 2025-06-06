#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Sub-package to display a GUI to build a simple Configuration object."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import param


def get_folder_icons() -> Path:
    import pyxel.gui.icons

    return Path(pyxel.gui.icons.__path__[0])


if TYPE_CHECKING:
    import panel as pn
    import xarray as xr

    from pyxel import Configuration


def display_header(text: str, doc: str | None = None) -> "pn.layout.Panel":
    """Create a header row for a Panel layout."""
    # Late import
    import panel as pn

    from pyxel.util import clean_text

    row = pn.Row(
        pn.widgets.StaticText(value=clean_text(text)),
        styles={"font-weight": "bold"},
        margin=(5, 10, 5, 0),
    )

    if doc:
        row.append(
            pn.widgets.TooltipIcon(
                value=clean_text(doc),
                margin=(0, 10, 0, -10),
            )
        )

    return row


class CCDGeometry(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    columns = param.Integer(100, bounds=(1, None))
    rows = param.Integer(100, bounds=(1, None))

    def view(self) -> "pn.layout.Panel":
        """Return a Panel layout to visualize the geometry fields."""
        # Late import
        import panel as pn

        from pyxel.util import get_schema

        schema = get_schema()["definitions"]["CCDGeometry"]

        return pn.Card(
            header=pn.Row(
                pn.pane.SVG(
                    get_folder_icons() / "adjustements_horizontal.svg",
                    width=18,
                    margin=(10, -5, 10, 0),
                    align="end",
                ),
                display_header(schema["description"]),
                margin=(0, 0, -10, 0),
            ),
            objects=[
                pn.widgets.IntInput.from_param(
                    self.param.columns,
                    description=schema["properties"]["col"]["description"],
                    margin=(5, 10, 5, 10),
                    sizing_mode="stretch_width",
                ),
                pn.widgets.IntInput.from_param(
                    self.param.rows,
                    description=schema["properties"]["row"]["description"],
                    margin=(5, 10, 10, 10),
                    sizing_mode="stretch_width",
                ),
            ],
            collapsible=False,
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=(10, 10, 5, 10),
        )


# class CMOSGeometry(param.Parameterized):
#     header = param.String("Geometrical attributes of a CMOS detector", doc="XXX")
#     columns = param.Integer(100, bounds=(1, None), doc="Number of pixel columns")
#     rows = param.Integer(100, bounds=(1, None), doc="Number of pixel rows")

#     def view(self):
#         return pn.Column(
#             display_header(self.param.header),
#             pn.widgets.IntInput.from_param(
#                 self.param.columns,
#                 margin=(5, 10, 5, 10),
#                 sizing_mode="stretch_width",
#             ),
#             pn.widgets.IntInput.from_param(
#                 self.param.rows,
#                 margin=(5, 10, 10, 10),
#                 sizing_mode="stretch_width",
#             ),
#             styles={"border": "2px solid black", "border_radius": "8px"},
#             margin=(10, 10, 5, 10),
#         )


class Environment(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    temperature = param.Number(
        default=None, bounds=(0, None), inclusive_bounds=(False, False)
    )

    def view(self) -> "pn.layout.Panel":
        """Return a Panel layout to visualize the geometry fields."""
        # Late import
        import panel as pn

        from pyxel.util import get_schema

        schema = get_schema()["definitions"]["Environment"]

        return pn.Card(
            header=pn.Row(
                pn.pane.SVG(
                    get_folder_icons() / "thermometer.svg",
                    width=18,
                    margin=(10, -5, 10, 0),
                    align="end",
                ),
                display_header(schema["description"]),
                margin=(0, 0, -10, 0),
            ),
            objects=[
                pn.widgets.FloatInput.from_param(
                    self.param.temperature,
                    step=1.0,
                    start=1e-6,
                    placeholder="Add a temperature ...",
                    description=schema["properties"]["temperature"]["description"],
                    margin=(5, 10, 10, 10),
                    sizing_mode="stretch_width",
                )
            ],
            collapsible=False,
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=(5, 10, 10, 10),
        )


class CCD(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    geometry = param.Parameter(CCDGeometry(), instantiate=True)
    environment = param.Parameter(Environment(), instantiate=True)

    def view(self) -> "pn.layout.Panel":
        """Return a Panel layout to visualize the geometry fields."""
        # Late import
        import panel as pn

        return pn.Card(
            header=pn.Row(
                pn.pane.SVG(
                    get_folder_icons() / "frame.svg",
                    width=18,
                    margin=(10, -5, 10, 0),
                    align="end",
                ),
                display_header("CCD Detector"),
                margin=(0, 0, -10, 0),
            ),
            objects=[self.geometry.view(), self.environment.view()],
            collapsible=False,
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=10,
        )


# class CMOS(param.Parameterized):
#     geometry = param.Parameter(CMOSGeometry(), instantiate=True)
#     environment = param.Parameter(Environment(), instantiate=True)

#     def view(self):
#         return pn.Column(
#             self.geometry.view(),
#             self.environment.view(),
#             styles={"border": "2px solid black", "border_radius": "8px"},
#             margin=10,
#         )

#
# class Readout(param.Parameterized):
#     """Configuration parameters and a Panel-based used interface."""
#
#     header = param.String("Readout mode", doc="XXX")
#     readout_time = param.Array(np.array([1.0, 2.0]), doc="Readout time")
#     non_destructive = param.Boolean(False, doc="Non-destructive readout mode")
#
#     def view(self) -> pn.layout.Panel:
#         """Return a Panel layout to visualize the geometry fields."""
#         return pn.Column(
#             display_param_header(self.param.header),
#             pn.Param(
#                 self.param.readout_time,
#                 sizing_mode="stretch_width",
#             ),
#             pn.Param(
#                 self.param.non_destructive,
#                 sizing_mode="stretch_width",
#             ),
#             styles={"border": "2px solid black", "border_radius": "8px"},
#             margin=10,
#         )


# TODO: Create a BaseClass or prototype
class ModelUSAF(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    enabled = param.Boolean(True)

    def view(self) -> "pn.layout.Panel":
        """Return a Panel layout to visualize the geometry fields."""
        # Late import
        import panel as pn

        return pn.Card(
            header=pn.Row(display_header("USAF illumination"), margin=(0, 0, -10, 0)),
            objects=[
                pn.Param(
                    self.param.enabled,
                    sizing_mode="stretch_width",
                )
            ],
            margin=10,
            # margin=(0, 0, -10, 0),
            styles={"border": "2px solid black", "border_radius": "8px"},
        )


# TODO: Create a BaseClass or prototype
class GroupPhotonCollection(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    models = param.List([ModelUSAF(name="usaf_illumination")], instantiate=True)

    def view(self) -> "pn.layout.Panel":
        """Return a Panel layout to visualize the geometry fields."""
        # Late import
        import panel as pn

        column = pn.Card(
            header=pn.Row(
                pn.pane.SVG(
                    get_folder_icons() / "layout_list.svg",
                    width=18,
                    margin=(10, -5, 10, 0),
                    align="end",
                ),
                display_header("Photon Collection group"),
                margin=(0, 0, -10, 0),
            ),
            collapsible=False,
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=(10, 10, 5, 10),
        )

        for el in self.models:
            column.append(el.view())

        return column


# TODO: Create a BaseClass or prototype
class ModelCDM(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    # Add more parameters
    enabled = param.Boolean(True)

    def view(self) -> "pn.layout.Panel":
        """Return a Panel layout to visualize the geometry fields."""
        # Late import
        import panel as pn

        return pn.Card(
            header=pn.Row(
                display_header("Charge Distortion Model"),
                margin=(0, 0, -10, 0),
            ),
            objects=[pn.Param(self.param.enabled, sizing_mode="stretch_width")],
            margin=10,
            styles={"border": "2px solid black", "border_radius": "8px"},
        )


# TODO: Create a BaseClass or prototype
class GroupChargeTransfer(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    models = param.List([ModelCDM(name="charge_distortion_model")], instantiate=True)

    def view(self) -> "pn.layout.Panel":
        """Return a Panel layout to visualize the geometry fields."""
        # Late import
        import panel as pn

        column = pn.Card(
            header=pn.Row(
                pn.pane.SVG(
                    get_folder_icons() / "layout_list.svg",
                    width=18,
                    margin=(10, -5, 10, 0),
                    align="end",
                ),
                display_header("Charge Transfer Collection group"),
                margin=(0, 0, -10, 0),
            ),
            collapsible=False,
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=10,
        )

        for el in self.models:
            column.append(el.view())

        return column


class CCDPipeline(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    photon_collection = param.Parameter()
    charge_transfer = param.Parameter()


# TODO: Add CMOSPipeline


class BasicConfigGUI(param.Parameterized):
    """Graphical User Interface for configuring and executing a basic detector pipeline.

    Examples
    --------
    >>> gui_config = BasicConfigGUI()
    >>> gui_config.display()
    """

    detector = param.Parameter(default=None)  # TODO: Improve this
    # readout = param.Parameter(Readout(), instantiate=True)
    pipeline = param.Parameter(default=None)  # TODO: Improve this

    def __init__(self, *args, **kwargs):
        # Late import
        import hvplot.xarray  # noqa: F401
        import panel as pn

        pn.extension("codeeditor")

        super().__init__(*args, **kwargs)

        ccd_param: param.Parameterized = CCD(name="CCD")  # TODO: Improve this
        # cmos_param: param.Parameterized = CMOS(name="CMOS")  # TODO: Improve this

        # ccd_param .view().visible = True
        # cmos_param.view().visble = False

        # TODO: Improve this
        self._detectors: Mapping[str, param.Parameterized] = {
            "CCD": ccd_param,
            # "CMOS": cmos_param,
        }

        # TODO: Improve this
        self._detectors_widget = {
            "CCD": ccd_param.view(),
            # "CMOS": cmos_param.view(),
        }

        # TODO: Improve this
        self.detector = self._detectors["CCD"]
        self._results_column = pn.Column()

        # photon_collection = GroupPhotonCollection()  # TODO: Improve this
        self._ccd_pipeline = CCDPipeline(
            photon_collection=GroupPhotonCollection(),
            charge_transfer=GroupChargeTransfer(),
        )
        self.pipeline = self._ccd_pipeline
        self._pipeline_column = pn.Card(
            header=pn.Row(
                pn.pane.SVG(
                    get_folder_icons() / "layers.svg",
                    width=18,
                    margin=(10, -5, 10, 0),
                    align="end",
                ),
                display_header("Pipeline / Models"),
                margin=(0, 0, -10, 0),
            ),
            collapsible=False,
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=(5, 10, 5, 10),
        )

        # self.pipeline.append(photon_collection)

    # def _select_detector(self, detector_name: str) -> None:
    #     # print(f"{detector_name=}, {self.detector=}")
    #     assert isinstance(detector_name, str)
    #     assert detector_name in self._detectors

    #     self._pipeline_column.clear()

    #     if detector_name == "CCD":  # TODO: Improve this
    #         self._detectors_widget["CCD"].visible = True
    #         self._detectors_widget["CMOS"].visible = False

    #         self._pipeline_column.append(self.pipeline.photon_collection.view())
    #         self._pipeline_column.append(self.pipeline.charge_transfer.view())

    #     else:
    #         self._detectors_widget["CCD"].visible = False
    #         self._detectors_widget["CMOS"].visible = True

    #         # self._pipeline_column.clear()
    #         self._pipeline_column.append(self.pipeline.photon_collection.view())
    #         # self._pipeline_column.append(self.pipeline.charge_transfer.view())

    #     self.detector = self._detectors[detector_name]

    def _run_pipeline(self, event):
        # Late import
        import panel as pn

        import pyxel
        from pyxel.gui import run_mode_gui

        # print(f"Run pipeline, {event=}")
        # TODO: Add a waiting status on the button
        self._results_column.clear()

        # Add Progress bar
        progress_widget = pn.widgets.Tqdm()
        self._results_column.append(progress_widget)

        config: "Configuration" = pyxel.build_configuration(
            self.detector.name,
            num_rows=self.detector.geometry.rows,  # TODO: use '.row' instead
            num_cols=self.detector.geometry.columns,  # TODO: use '.column' instead
        )

        config.running_mode.readout.times = [1.0, 2.0, 3.0]
        config.running_mode.readout.non_destructive = True

        assert config.pipeline.photon_collection
        config.pipeline.photon_collection.usaf_illumination.enabled = (
            self.pipeline.photon_collection.models[0].enabled
        )

        with self._button.param.update(loading=True):
            self._button.loading = True
            ds: xr.Dataset = run_mode_gui(config, tqdm_widget=progress_widget)
            image_tabs = pyxel.display_dataset(ds, orientation="vertical")

        # Python snippet code
        source_code_lst: list[str] = []
        source_code_lst.append("# Snippet code")
        source_code_lst.append("import pyxel")
        source_code_lst.append("")
        source_code_lst.append(f"# Get default {self.detector.name!r} pipeline")
        source_code_lst.append(
            f"config = pyxel.build_configuration({self.detector.name!r}, num_cols={self.detector.geometry.columns!r}, num_rows={self.detector.geometry.rows!r})"
        )
        source_code_lst.append("")

        if self.detector.environment.temperature:
            source_code_lst.append(
                f"config.detector.environment.temperature = {self.detector.environment.temperature!r}"
            )

        source_code_lst.append(
            f"config.running_mode.readout.times = {config.running_mode.readout.times.tolist()!r}"
        )
        source_code_lst.append(
            f"config.running_mode.readout.non_destructive = {config.running_mode.readout.non_destructive!r}"
        )
        source_code_lst.append(
            f"config.pipeline.photon_collection.usaf_illumination.enabled = {config.pipeline.photon_collection.usaf_illumination.enabled!r}"
        )
        source_code_lst.append("")
        source_code_lst.append(
            f"# pyxel.save(config, filename='demo_{self.detector.name}.yaml')  # Not yet implemented"
        )
        source_code_lst.append("")
        source_code_lst.append("result = pyxel.run_mode_dataset(config)")
        source_code_lst.append("")
        source_code_lst.append("pyxel.display_detector(config.detector)")

        source_code = "\n".join(source_code_lst)

        code_editor_widget = pn.widgets.CodeEditor(
            value=source_code,
            name="Code",
            language="python",
            sizing_mode="stretch_width",
            readonly=True,
        )

        yaml_widget = pn.widgets.CodeEditor(
            value=config.to_yaml(),
            name="Configuration",
            language="yaml",
            sizing_mode="stretch_width",
            readonly=True,
        )

        image_tabs.append(code_editor_widget)
        image_tabs.append(yaml_widget)

        self._results_column.append(image_tabs)

    def display(self):
        # Select Detector
        # widget_detectors = pn.widgets.RadioButtonGroup(
        #     value="CCD",  # TODO: Improve this
        #     options=["CCD", "CMOS"],  # TODO: Improve this
        #     button_type="primary",
        #     button_style="outline",
        #     sizing_mode="stretch_width",
        # )
        # iref = pn.bind(self._select_detector, detector_name=widget_detectors)
        # Late import
        import panel as pn

        config_column = pn.Column(
            width=400,
            margin=5,
            styles={
                "background": "WhiteSmoke",
                "border": "2px solid black",
                # "border": "1px solid #dee2e6",
                # "border": "1px solid black",
                # 'box_shadow':'0 1px 3px rgba(0 ,0, 0, 0.1)',
                "border_radius": "8px",
            },
        )
        config_column.append(
            pn.pane.Markdown(
                "### Pre-defined Configuration",
                align="center",
                margin=(5, 0, 0, 0),
            )
        )
        # column.append(widget_detectors)
        # column.append(iref)

        self._detectors_widget["CCD"].visible = True
        # self._detectors_widget["CMOS"].visible = False

        config_column.append(self._detectors_widget["CCD"])  # TODO: Improve this
        # column.append(self._detectors_widget["CMOS"])  # TODO: Improve this

        # Readout
        # column.append(self.readout.view())

        # Model groups
        # self._pipeline_column = pn.Column(styles={"border": "2px solid black"}, margin=10)

        # TODO: Improve this
        # Only for CCDs
        self._pipeline_column.append(self.pipeline.photon_collection.view())
        self._pipeline_column.append(self.pipeline.charge_transfer.view())

        config_column.append(self._pipeline_column)

        self._button = pn.widgets.Button(
            name="Execute pipeline",
            on_click=self._run_pipeline,
            button_type="primary",
            sizing_mode="stretch_width",
        )
        config_column.append(self._button)

        return pn.Row(
            config_column,
            self._results_column,
            width=400 * 3,
            styles={
                "background": "WhiteSmoke",
                # "border": "2px dashed yellow",
            },
        )


# if __name__ == '__main__':
# pip install watchfiles
#
# panel serve simple_config.py --dev --show
# config = PredefinedConfig()
# config.display().servable(title="Pyxel")
