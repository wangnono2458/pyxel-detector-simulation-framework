#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import pyxel
from pyxel import Configuration


def test_loads(valid_minimalist_exposure_config: str):
    """Test function 'pyxel.loads'."""
    content: str = valid_minimalist_exposure_config
    assert isinstance(content, str)

    cfg = pyxel.loads(content)
    assert isinstance(cfg, Configuration)
