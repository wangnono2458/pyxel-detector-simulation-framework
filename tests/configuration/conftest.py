#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import pytest


@pytest.fixture(
    scope="session",
    params=["ccd_detector", "cmos_detector", "apd_detector", "mkid_detector"],
)
def valid_minimalist_exposure_config(request) -> str:
    """Minimalist Exposure config."""
    detector_name: str = request.param
    assert isinstance(detector_name, str)

    if detector_name in ["ccd_detector", "cmos_detector", "mkid_detector"]:
        return f"""# Add unicode: µm, e⁻
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
    elif detector_name == "apd_detector":
        return """# Add unicode: µm, e⁻
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
    else:
        raise NotImplementedError
