#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import pytest

from pyxel.configuration.configuration import _add_comments


@pytest.mark.parametrize(
    "text, exp_text",
    [
        pytest.param(
            "  environment:\n    temperature: 12",
            "  environment:\n    temperature: 12  # Unit: [K]",
            id="Temperature, int",
        ),
        pytest.param(
            "  environment:\n    temperature: 3.14",
            "  environment:\n    temperature: 3.14  # Unit: [K]",
            id="Temperature, float",
        ),
        pytest.param(
            "  environment:\n    temperature: null",
            "  environment:\n    temperature: null  # Unit: [K]",
            id="Temperature, null",
        ),
        pytest.param("row:", "row:  # Unit: [pix]", id="row"),
        pytest.param("col:", "col:  # Unit: [pix]", id="col"),
        pytest.param("times:", "times:  # Unit: [s]", id="times"),
        pytest.param(
            "exposure\n  readout:\n    times:",
            "exposure\n  readout:\n    times:  # Unit: [s]",
            id="times, multi lines",
        ),
        pytest.param("start_time:", "start_time:  # Unit: [s]", id="start_time"),
        pytest.param(
            "total_thickness:", "total_thickness:  # Unit: [µm]", id="total_thickness"
        ),
        pytest.param(
            "pixel_vert_size:", "pixel_vert_size:  # Unit: [µm]", id="pixel_vert_size"
        ),
        pytest.param(
            "pixel_horz_size:", "pixel_horz_size:  # Unit: [µm]", id="pixel_horz_size"
        ),
        pytest.param(
            "pixel_scale:", "pixel_scale:  # Unit: [arcsec / pix]", id="pixel_scale"
        ),
        pytest.param(
            "charge_to_volt_conversion:",
            "charge_to_volt_conversion:  # Unit: [V / electron]",
            id="charge_to_volt_conversion",
        ),
        pytest.param(
            "pre_amplification:",
            "pre_amplification:  # Unit: [V / V]",
            id="pre_amplification",
        ),
        pytest.param(
            "adc_voltage_range:",
            "adc_voltage_range:  # Unit: [V]",
            id="adc_voltage_range",
        ),
        pytest.param(
            "adc_bit_resolution:",
            "adc_bit_resolution:  # Unit: [bit]",
            id="adc_bit_resolution",
        ),
        pytest.param(
            "full_well_capacity:",
            "full_well_capacity:  # Unit: [electron]",
            id="full_well_capacity",
        ),
    ],
)
def test_add_comments(text: str, exp_text: str):
    """Test function '_add_comments'."""
    result = _add_comments(text)

    assert result == exp_text
