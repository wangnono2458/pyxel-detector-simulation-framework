#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from collections.abc import Mapping

import numpy as np
import panel as pn
import param
import xarray as xr

import pyxel
from pyxel import Configuration
from pyxel.gui import run_mode_gui

pn.extension("codeeditor")


def display_header(obj: param.String) -> pn.layout.Panel:
    """Create a header row for a Panel layout."""
    return pn.Row(
        pn.widgets.StaticText(value=obj.default),
        pn.widgets.TooltipIcon(value=obj.doc, margin=(0, 10, 0, -10)),
        styles={"font-weight": "bold"},
        margin=(5, 10, 5, 0),
    )


class CCDGeometry(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    header = param.String("Geometrical attributes of a CCD detector", doc="XXX")
    columns = param.Integer(100, bounds=(1, None), doc="Number of pixel columns")
    rows = param.Integer(100, bounds=(1, None), doc="Number of pixel rows")

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        return pn.Column(
            display_header(self.param.header),
            pn.widgets.IntInput.from_param(
                self.param.columns,
                margin=(5, 10, 5, 10),
                sizing_mode="stretch_width",
            ),
            pn.widgets.IntInput.from_param(
                self.param.rows,
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

    header = param.String("Environmental attributes of the detector", doc="XXX")
    temperature = param.Number(
        default=None,
        bounds=(0, None),
        inclusive_bounds=(False, False),
        doc="XXX",
    )

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        return pn.Column(
            display_header(self.param.header),
            pn.widgets.FloatInput.from_param(
                self.param.temperature,
                step=1.0,
                start=1e-6,
                placeholder="Add a temperature ...",
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


# TODO: Create a BaseClass or prototype
class ModelUSAF(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    header = param.String("USAF illumination", doc="XXX")
    enabled = param.Boolean(True)

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        return pn.Card(
            header=display_header(self.param.header),
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

    header = param.String("Photon Collection group", doc="XXX")
    models = param.List([ModelUSAF(name="usaf_illumination")], instantiate=True)

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        column = pn.Column(
            display_header(self.param.header),
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=10,
        )

        for el in self.models:
            column.append(el.view())

        return column


# TODO: Create a BaseClass or prototype
class ModelCDM(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    header = param.String("Charge Distortion Model", doc="XXX")
    # Add more parameters
    enabled = param.Boolean(True)

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        return pn.Card(
            header=display_header(self.param.header),
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

    header = param.String("Charge Transfer Collection group", doc="XXX")
    models = param.List([ModelCDM(name="charge_distortion_model")], instantiate=True)

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        column = pn.Column(
            display_header(self.param.header),
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=10,
        )

        for el in self.models:
            column.append(el.view())

        return column


class Readout(param.Parameterized):
    """Configuration parameters and a Panel-based used interface."""

    header = param.String("Readout mode", doc="XXX")
    readout_time = param.Array(np.array([1.0, 2.0]), doc="Readout time")
    non_destructive = param.Boolean(False, doc="Non-destructive readout mode")

    def view(self) -> pn.layout.Panel:
        """Return a Panel layout to visualize the geometry fields."""
        return pn.Column(
            display_header(self.param.header),
            pn.Param(
                self.param.readout_time,
                sizing_mode="stretch_width",
            ),
            pn.Param(
                self.param.non_destructive,
                sizing_mode="stretch_width",
            ),
            styles={"border": "2px solid black", "border_radius": "8px"},
            margin=10,
        )


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
    readout = param.Parameter(Readout(), instantiate=True)
    pipeline = param.Parameter(default=None)  # TODO: Improve this
    # photon_collection = param.Parameter(group_photon_collection_param)# TODO: Remove this

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
            styles={"border": "2px solid black"},
            margin=10,
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
            ds: xr.Dataset = run_mode_gui(config, tqdm_widget=progress_widget)
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

        column = pn.Column(
            width=400,
            styles={
                "background": "WhiteSmoke",
                "border": "2px solid black",
                # "border": "1px solid #dee2e6",
                # "border": "1px solid black",
                # 'box_shadow':'0 1px 3px rgba(0 ,0, 0, 0.1)',
                "border_radius": "8px",
            },
        )
        column.append(
            pn.pane.Markdown("### Pre-defined Configuration", align="center", margin=0)
        )
        # column.append(widget_detectors)
        # column.append(iref)

        self._detectors_widget["CCD"].visible = True
        # self._detectors_widget["CMOS"].visible = False

        column.append(self._detectors_widget["CCD"])  # TODO: Improve this
        # column.append(self._detectors_widget["CMOS"])  # TODO: Improve this

        # Readout
        column.append(self.readout.view())

        # Model groups
        # self._pipeline_column = pn.Column(styles={"border": "2px solid black"}, margin=10)

        # TODO: Improve this
        # Only for CCDs
        self._pipeline_column.append(self.pipeline.photon_collection.view())
        self._pipeline_column.append(self.pipeline.charge_transfer.view())

        column.append(self._pipeline_column)

        self._button = pn.widgets.Button(
            name="Execute pipeline",
            on_click=self._run_pipeline,
            button_type="primary",
            sizing_mode="stretch_width",
        )
        column.append(self._button)

        return pn.Row(
            column,
            self._results_column,
            width=400 * 3,
            styles={
                "background": "WhiteSmoke",
                # "border": "2px dashed yellow",
            },
        )
