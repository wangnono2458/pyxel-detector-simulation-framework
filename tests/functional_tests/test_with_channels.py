#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path

import pytest
import xarray as xr

import pyxel


@pytest.fixture(
    params=[
        "ccd_detector",
        "cmos_detector",
        # 'apd_detector',
    ]
)
def content_exposure_with_channels(request) -> str:
    """Create YAML exposure content."""
    detector: str = request.param
    return f"""
exposure:
  readout:
    times: [1]

{detector}:

  geometry:
    row: 100               # pixel
    col: 100               # pixel

    channels:
      matrix: [[OP9, OP13],
               [OP1, OP5 ]]
      readout_position:
        OP9:  top-left
        OP13: top-left
        OP1:  bottom-left
        OP5:  bottom-left

  environment:
    temperature: 100        # K
  characteristics:
    quantum_efficiency: 1.0                # - for charge_generation.simple_conversion
    charge_to_volt:
      value: 1                             # V/e for charge_measurement.simple_measurement
    pre_amplification: 1                 # V/V  for readout_electronics.simple_amplifier
    adc_bit_resolution: 16        # for readout_electronics.simple_adc
    adc_voltage_range: [0.,65535.0]     # for readout_electronics.simple_adc

pipeline:

  photon_collection:
    # -> photon
    - name: illumination
      func: pyxel.models.photon_collection.illumination
      arguments:
        level: 1000

  charge_generation:
    # photon -> charge
    - name: simple_conversion
      func: pyxel.models.charge_generation.simple_conversion

  charge_collection:
    # charge -> pixel
    - name: simple_collection
      func: pyxel.models.charge_collection.simple_collection

  charge_measurement:
    # pixel -> signal
    - name: simple_measurement
      func: pyxel.models.charge_measurement.simple_measurement
      enabled: true

    # signal -> signal
    - name: dc_crosstalk
      func: pyxel.models.charge_measurement.dc_crosstalk
      arguments:
        coupling_matrix: [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]]
#        channel_matrix: [1,2,3,4]
#        readout_directions: [1,2,1,2]

  readout_electronics:
    # -> signal
    - name: simple_amplifier
      func: pyxel.models.readout_electronics.simple_amplifier

    # signal -> image
    - name: simple_adc
      func: pyxel.models.readout_electronics.simple_adc
    """


@pytest.fixture
def filename_with_channels(tmp_path: Path, content_exposure_with_channels: str) -> Path:
    """Get YAML filename."""
    filename: Path = tmp_path / "exposure_with_channels.yaml"

    filename.write_text(content_exposure_with_channels)
    return filename


def test_exposure_with_channels(filename_with_channels: Path):
    """Test an exposure YAML file with Channels."""
    cfg = pyxel.load(filename_with_channels)

    result = pyxel.run_mode(cfg)
    assert isinstance(result, xr.DataTree)
