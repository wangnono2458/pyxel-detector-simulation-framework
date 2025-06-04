#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Sub-package to display a GUI to build a simple Configuration object."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import panel as pn
import param

import pyxel
from pyxel import Configuration
from pyxel.gui import run_mode_gui
from pyxel.util import clean_text, get_schema

pn.extension("codeeditor")

if TYPE_CHECKING:
    import xarray as xr


def display_header(text: str, doc: str | None = None) -> pn.layout.Panel:
    """Create a header row for a Panel layout."""
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

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        schema = get_schema()["definitions"]["CCDGeometry"]

        return pn.Column(
            display_header(schema["description"]),
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

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        schema = get_schema()["definitions"]["Environment"]

        return pn.Column(
            display_header(schema["description"]),
            pn.widgets.FloatInput.from_param(
                self.param.temperature,
                step=1.0,
                start=1e-6,
                placeholder="Add a temperature ...",
                description=schema["properties"]["temperature"]["description"],
                margin=(5, 10, 10, 10),
                sizing_mode="stretch_width",
            ),
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=(5, 10, 10, 10),
        )


class CCD(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    geometry = param.Parameter(CCDGeometry(), instantiate=True)
    environment = param.Parameter(Environment(), instantiate=True)

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        return pn.Column(
            # display_header('CCD Detector'),
            self.geometry.view(),
            self.environment.view(),
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

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        return pn.Card(
            header=display_header("USAF illumination"),
            objects=[
                pn.Param(
                    self.param.enabled,
                    sizing_mode="stretch_width",
                )
            ],
            margin=10,
            styles={"border": "2px solid black", "border_radius": "8px"},
        )


# TODO: Create a BaseClass or prototype
class GroupPhotonCollection(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    models = param.List([ModelUSAF(name="usaf_illumination")], instantiate=True)

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        column = pn.Column(
            display_header("Photon Collection group"),
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

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        return pn.Card(
            header=display_header("Charge Distortion Model"),
            objects=[
                pn.Param(
                    self.param.enabled,
                    sizing_mode="stretch_width",
                )
            ],
            margin=10,
            styles={"border": "2px solid black", "border_radius": "8px"},
        )


# TODO: Create a BaseClass or prototype
class GroupChargeTransfer(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    models = param.List([ModelCDM(name="charge_distortion_model")], instantiate=True)

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        column = pn.Column(
            display_header("Charge Transfer Collection group"),
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


class PredefinedConfig(param.Parameterized):
    """Pre-defined detector pipeline with Panel UI.

    Examples
    --------
    >>> gui_config = PredefinedConfig()
    >>> gui_config.display()
    """

    detector = param.Parameter(default=None)  # TODO: Improve this
    # readout = param.Parameter(Readout(), instantiate=True)
    pipeline = param.Parameter(default=None)  # TODO: Improve this

    def __init__(self, *args, **kwargs):
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
        self._pipeline_column = pn.Column(
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
        # print(f"Run pipeline, {event=}")
        # TODO: Add a waiting status on the button
        self._results_column.clear()

        # Add Progress bar
        progress_widget = pn.widgets.Tqdm()
        self._results_column.append(progress_widget)

        config: Configuration = pyxel.build_configuration(
            self.detector.name,
            num_rows=self.detector.geometry.rows,  # TODO: use '.row' instead
            num_cols=self.detector.geometry.columns,  # TODO: use '.column' instead
        )

        config.running_mode.readout.times = self.readout.readout_time.tolist()
        config.running_mode.readout.non_destructive = self.readout.non_destructive

        assert config.pipeline.photon_collection
        config.pipeline.photon_collection.usaf_illumination.enabled = (
            self.pipeline.photon_collection.models[0].enabled
        )

        with self._button.param.update(loading=True):
            self._button.loading = True
            _ds: xr.Dataset = run_mode_gui(config, tqdm_widget=progress_widget)
            image_tabs = pyxel.display_detector(config.detector)

        # Python snippet code
        source_code: str = f"""# Snippet code
import pyxel

config = pyxel.build_configuration({self.detector.name!r}, num_cols={self.detector.geometry.columns!r}, num_rows={self.detector.geometry.rows!r})  # Get default {self.detector.name!r} pipeline

config.detector.environment.temperature = {self.detector.environment.temperature!r}
config.running_mode.readout.times = {config.running_mode.readout.times.tolist()!r}
config.running_mode.readout.non_destructive = {config.running_mode.readout.non_destructive!r}
config.pipeline.photon_collection.usaf_illumination.enabled = {config.pipeline.photon_collection.usaf_illumination.enabled!r}

# pyxel.save(config, filename='demo_{self.detector.name}.yaml')  # Not yet implemented

result = pyxel.run_mode_dataset(config)

pyxel.display_detector(config.detector)
"""

        code_editor_widget = pn.widgets.CodeEditor(
            value=source_code,
            name="Code",
            sizing_mode="stretch_width",
            language="python",
            readonly=True,
        )

        image_tabs.append(code_editor_widget)
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


# # pip install watchfiles
# # panel serve simple_config.py --dev --show
# foo = PredefinedConfig()
# foo.display().servable(title="Pyxel")
# pn.serve(foo.display(), dev=True)
