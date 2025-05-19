#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from pathlib import Path

import pytest

import pyxel
from pyxel import Configuration

# This is equivalent to 'import freezegun'
freezegun = pytest.importorskip(
    "freezegun",
    reason="Package 'freezegun' is not installed. Use 'pip install freezegun'",
)


@pytest.fixture
def config_no_running_mode_deprecated(tmp_path: Path) -> Path:
    """Create a configuration file without a valid running mode."""
    content = """
ccd_detector:

  geometry:

    row: 450               # pixel
    col: 450               # pixel
    total_thickness: 40.    # um
    pixel_vert_size: 10.    # um
    pixel_horz_size: 10.    # um

  environment:
    temperature: 300        # K

  characteristics:
    qe:   1.                # -
    eta:  1.                # e/photon
    sv:   3.e-6             # V/e
    amp:  10.               # V/V
    a1:   100.              # V/V
    a2:   3000              # DN/V
    fwc:  2000              # e
    fwc_serial: 10000       # e
    vg:   1.62e-10          # cm^2
    svg:  1.62e-10          # cm^2
    t:    9.4722e-04        # s
    st:   9.4722e-04        # s

pipeline:
  photon_generation:
    - name: illumination
      func: pyxel.models.photon_generation.illumination
      enabled: true
      arguments:
          level: 0
  optics:
  charge_generation:
  charge_collection:
  charge_transfer:
  charge_measurement:
  readout_electronics:
"""

    filename = tmp_path / "no_running_mode.yaml"

    with filename.open(mode="w") as fh:
        fh.write(content)

    return filename


@pytest.fixture
def config_no_running_mode(tmp_path: Path) -> Path:
    """Create a configuration file without a valid running mode."""
    content = """
ccd_detector:

  geometry:

    row: 450               # pixel
    col: 450               # pixel
    total_thickness: 40.    # um
    pixel_vert_size: 10.    # um
    pixel_horz_size: 10.    # um

  environment:
    temperature: 300        # K

  characteristics:
    qe:   1.                # -
    eta:  1.                # e/photon
    sv:   3.e-6             # V/e
    amp:  10.               # V/V
    a1:   100.              # V/V
    a2:   3000              # DN/V
    fwc:  2000              # e
    fwc_serial: 10000       # e
    vg:   1.62e-10          # cm^2
    svg:  1.62e-10          # cm^2
    t:    9.4722e-04        # s
    st:   9.4722e-04        # s

pipeline:
  photon_collection:
    - name: illumination
      func: pyxel.models.photon_collection.illumination
      enabled: true
      arguments:
          level: 0
  charge_generation:
  charge_collection:
  charge_transfer:
  charge_measurement:
  readout_electronics:
"""

    filename = tmp_path / "no_running_mode.yaml"

    with filename.open(mode="w") as fh:
        fh.write(content)

    return filename


@pytest.fixture
def config_two_running_modes_deprecated(tmp_path: Path) -> Path:
    """Create a configuration file with two valid running mode."""
    content = """
exposure:

  outputs:
    output_folder: "output"
    save_data_to_file:
      - detector.image.array:   ['fits']
      - detector.pixel.array: ['npy']

observation:

  parameters:
    - key: pipeline.photon_generation.illumination.arguments.level
      values: numpy.unique(numpy.logspace(0, 6, 10, dtype=int))

  outputs:
    output_folder:  'outputs'
    # each pipeline run
    save_data_to_file:
      - detector.image.array:   ['npy']
    # once at the end
    save_observation_data:


ccd_detector:

  geometry:

    row: 450               # pixel
    col: 450               # pixel
    total_thickness: 40.    # um
    pixel_vert_size: 10.    # um
    pixel_horz_size: 10.    # um

  environment:
    temperature: 300        # K

  characteristics:
    qe:   1.                # -
    eta:  1.                # e/photon
    sv:   3.e-6             # V/e
    amp:  10.               # V/V
    a1:   100.              # V/V
    a2:   3000              # DN/V
    fwc:  2000              # e
    fwc_serial: 10000       # e
    vg:   1.62e-10          # cm^2
    svg:  1.62e-10          # cm^2
    t:    9.4722e-04        # s
    st:   9.4722e-04        # s

pipeline:
  photon_generation:
    - name: illumination
      func: pyxel.models.photon_generation.illumination
      enabled: true
      arguments:
          level: 0
  optics:
  charge_generation:
  charge_collection:
  charge_transfer:
  charge_measurement:
  readout_electronics:
"""

    filename = tmp_path / "no_running_mode.yaml"

    with filename.open(mode="w") as fh:
        fh.write(content)

    return filename


@pytest.fixture
def config_two_running_modes(tmp_path: Path) -> Path:
    """Create a configuration file with two valid running mode."""
    content = """
exposure:

  outputs:
    output_folder: "output"
    save_data_to_file:
      - detector.image.array:   ['fits']
      - detector.pixel.array: ['npy']

observation:

  parameters:
    - key: pipeline.photon_collection.illumination.arguments.level
      values: numpy.unique(numpy.logspace(0, 6, 10, dtype=int))

  outputs:
    output_folder:  'outputs'
    # each pipeline run
    save_data_to_file:
      - detector.image.array:   ['npy']
    # once at the end
    save_observation_data:


ccd_detector:

  geometry:

    row: 450               # pixel
    col: 450               # pixel
    total_thickness: 40.    # um
    pixel_vert_size: 10.    # um
    pixel_horz_size: 10.    # um

  environment:
    temperature: 300        # K

  characteristics:
    qe:   1.                # -
    eta:  1.                # e/photon
    sv:   3.e-6             # V/e
    amp:  10.               # V/V
    a1:   100.              # V/V
    a2:   3000              # DN/V
    fwc:  2000              # e
    fwc_serial: 10000       # e
    vg:   1.62e-10          # cm^2
    svg:  1.62e-10          # cm^2
    t:    9.4722e-04        # s
    st:   9.4722e-04        # s

pipeline:
  photon_collection:
    - name: illumination
      func: pyxel.models.photon_collection.illumination
      enabled: true
      arguments:
          level: 0
  charge_generation:
  charge_collection:
  charge_transfer:
  charge_measurement:
  readout_electronics:
"""

    filename = tmp_path / "no_running_mode.yaml"

    with filename.open(mode="w") as fh:
        fh.write(content)

    return filename


@pytest.fixture
def config_no_detector(tmp_path: Path) -> Path:
    """Create a configuration file without detector."""
    content = """
exposure:

  outputs:
    output_folder: "output"
    save_data_to_file:
      - detector.image.array:   ['fits']
      - detector.pixel.array: ['npy']

pipeline:
  photon_collection:
    - name: illumination
      func: pyxel.models.photon_collection.illumination
      enabled: true
      arguments:
          level: 0
  charge_generation:
  charge_collection:
  charge_transfer:
  charge_measurement:
  readout_electronics:
"""

    filename = tmp_path / "no_running_mode.yaml"

    with filename.open(mode="w") as fh:
        fh.write(content)

    return filename


@pytest.fixture
def config_two_detectors(tmp_path: Path) -> Path:
    """Create a configuration file with two detectors."""
    content = """
ccd_detector:

  geometry:

    row: 450               # pixel
    col: 450               # pixel
    total_thickness: 40.    # um
    pixel_vert_size: 10.    # um
    pixel_horz_size: 10.    # um

  environment:
    temperature: 300        # K

  characteristics:
    qe:   1.                # -
    eta:  1.                # e/photon
    sv:   3.e-6             # V/e
    amp:  10.               # V/V
    a1:   100.              # V/V
    a2:   3000              # DN/V
    fwc:  2000              # e
    fwc_serial: 10000       # e
    vg:   1.62e-10          # cm^2
    svg:  1.62e-10          # cm^2
    t:    9.4722e-04        # s
    st:   9.4722e-04        # s

cmos_detector:

  geometry:

    row: 100               # pixel
    col: 100               # pixel
    total_thickness: 40.    # um
    pixel_vert_size: 10.    # um
    pixel_horz_size: 10.    # um
    n_output: 1
    n_row_overhead: 0
    n_frame_overhead: 0
    reverse_scan_direction: False
    reference_pixel_border_width: 4

  environment:
    temperature: 300

  characteristics:
    qe: 0.5               # -
    eta: 1                # e/photon
    sv: 1.0e-6            # V/e
    amp: 0.8              # V/V
    a1: 100               # V/V
    a2: 50000             # DN/V
    #a2: 65536             # DN/V
    fwc: 100000            # e
    dsub: 0.5

exposure:

  outputs:
    output_folder: "output"
    save_data_to_file:
      - detector.image.array:   ['fits']
      - detector.pixel.array: ['npy']

pipeline:
  photon_collection:
    - name: illumination
      func: pyxel.models.photon_collection.illumination
      enabled: true
      arguments:
          level: 0
  charge_generation:
  charge_collection:
  charge_transfer:
  charge_measurement:
  readout_electronics:
"""

    filename = tmp_path / "no_running_mode.yaml"

    with filename.open(mode="w") as fh:
        fh.write(content)

    return filename


@pytest.fixture
def folder_data(request: pytest.FixtureRequest) -> Path:
    """Get the folder 'tests'."""
    folder = Path(request.module.__file__).parent
    filename: Path = folder / "../data"
    return filename.resolve(strict=True)


def test_load_2_times(folder_data: Path):
    """Test function 'pyxel.load' called two times."""
    # Get full filename
    full_filename: Path = folder_data / "dummy_simple.yaml"
    assert full_filename.exists()

    # Load the configuration file for the first time
    with freezegun.freeze_time("2021-06-15 14:11"):
        _ = pyxel.load(full_filename)

    # Load the configuration file for the second time
    with freezegun.freeze_time("2021-06-15 14:11"):
        _ = pyxel.load(full_filename)


def test_load_no_running_mode(config_no_running_mode: Path):
    """Test function 'pyxel.load' without a running mode."""
    filename = config_no_running_mode

    with pytest.raises(
        ValueError,
        match=(
            r"Expecting only one running mode: 'exposure', 'observation', 'calibration'"
        ),
    ):
        _ = pyxel.load(filename)


def test_load_two_running_modes(config_two_running_modes: Path):
    """Test function 'pyxel.load' without two running modes."""
    filename = config_two_running_modes

    with pytest.raises(
        ValueError,
        match=(
            r"Expecting only one running mode: 'exposure', 'observation', 'calibration'"
        ),
    ):
        _ = pyxel.load(filename)


def test_load_no_detector(config_no_detector: Path):
    """Test function 'pyxel.load' without detector."""
    filename = config_no_detector

    with pytest.raises(
        ValueError,
        match=(
            r"Got no detector. Expected values: 'ccd_detector', 'cmos_detector',"
            r" 'mkid_detector'"
        ),
    ):
        _ = pyxel.load(filename)


def test_load_two_detectors(config_two_detectors: Path):
    """Test function 'pyxel.load' with two detectors."""
    filename = config_two_detectors

    with pytest.raises(
        ValueError,
        match=(
            r"Expecting only one detector, got 2 detectors. "
            r"Expected values: 'ccd_detector', 'cmos_detector', 'mkid_detector'"
        ),
    ):
        _ = pyxel.load(filename)


def test_load_observation_without_parameters(tmp_path: Path):
    """Test function 'pyxel.load' with Observation mode but without 'parameters'."""
    content = """
observation:
  readout:
    times: [10, 20, 30]
    non_destructive: true

  # parameters:
  #   - key: pipeline.photon_collection.illumination.arguments.level
  #     values: numpy.unique(numpy.logspace(0, 6, 10, dtype=int))
  #
cmos_detector:
  geometry:
    row: 16
    col: 16
  environment:
  characteristics:
    charge_to_volt_conversion: 3.0e-6
    pre_amplification: 100
    adc_bit_resolution: 16
    adc_voltage_range: [0.,6.]

pipeline:
  photon_collection:
  charge_generation:
  charge_collection:
  charge_measurement:
  readout_electronics:
  data_processing:
"""

    filename = tmp_path / "observation_no_parameters.yaml"

    with filename.open(mode="w") as fh:
        fh.write(content)

    with pytest.raises(
        ValueError, match=r"Missing entry \'parameters\' in the YAML configuration file"
    ):
        pyxel.load(filename)


@pytest.mark.parametrize(
    "detector_name",
    ["ccd_detector", "cmos_detector", "mkid_detector"],
)
def test_load_exposure_minimalist_config(detector_name: str, tmp_path: Path):
    """Test function 'pyxel.load' with a minimalist YAML file."""
    content = f"""
exposure:
  readout:
    times: [1., 3., 5.]
    non_destructive:  true

    # No 'outputs'.

{detector_name}:
  geometry:
    row: 10
    col: 20

  # No 'environment' and 'characteristics'
"""

    filename = tmp_path / "exposure_minimalist.yaml"
    filename.write_text(content)

    cfg = pyxel.load(filename)
    assert isinstance(cfg, Configuration)


def test_load_exposure_minimalist_config_apd(tmp_path: Path):
    """Test function 'pyxel.load' with a minimalist YAML file."""
    content = """
exposure:
  readout:
    times: [1., 3., 5.]
    non_destructive:  true

    # No 'outputs'.

apd_detector:
  geometry:
    row: 10
    col: 20

  characteristics:
    roic_gain: 1.0
    avalanche_gain: 1.0
    common_voltage: 0.0

  # No 'environment'
"""

    filename = tmp_path / "exposure_minimalist.yaml"
    filename.write_text(content)

    cfg = pyxel.load(filename)
    assert isinstance(cfg, Configuration)


@pytest.mark.parametrize(
    "detector_name", ["ccd_detector", "cmos_detector", "mkid_detector"]
)
def test_load_exposure_minimalist_config_empty_outputs(
    detector_name: str, tmp_path: Path
):
    """Test function 'pyxel.load' with a minimalist YAML file."""
    content = f"""
exposure:
  readout:
    times: [1., 3., 5.]
    non_destructive:  true

  outputs:  # empty

{detector_name}:
  geometry:
    row: 10
    col: 20

  # No 'environment' and 'characteristics'
"""

    filename = tmp_path / "exposure_minimalist.yaml"
    filename.write_text(content)

    cfg = pyxel.load(filename)
    assert isinstance(cfg, Configuration)


def test_load_exposure_minimalist_config_empty_outputs_apd(tmp_path: Path):
    """Test function 'pyxel.load' with a minimalist YAML file."""
    content = """
exposure:
  readout:
    times: [1., 3., 5.]
    non_destructive:  true

  outputs:  # empty

apd_detector:
  geometry:
    row: 10
    col: 20

  characteristics:
    roic_gain: 1.0
    avalanche_gain: 1.0
    common_voltage: 0.0

  # No 'environment'
"""

    filename = tmp_path / "exposure_minimalist.yaml"
    filename.write_text(content)

    cfg = pyxel.load(filename)
    assert isinstance(cfg, Configuration)
