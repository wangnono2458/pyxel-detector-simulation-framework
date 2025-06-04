#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import pytest

from pyxel.util import clean_text, get_schema


@pytest.mark.parametrize(
    "content, exp_result",
    [
        pytest.param("Hello\nworld", "Hello\nworld", id="No modifications"),
        pytest.param(
            "Geometrical attributes of a :term:`CCD` detector.",
            "Geometrical attributes of a CCD detector.",
            id="One 'term'",
        ),
        pytest.param(
            "You can check the arguments specification in :ref:`Load image`",
            "You can check the arguments specification in 'Load image'",
            id="one 'ref'",
        ),
    ],
)
def test_clean_text(content: str, exp_result: str):
    """Test function 'clean_text'."""
    result = clean_text(content)
    assert result == exp_result


def test_get_schema():
    """Test function 'get_schema'."""
    dct = get_schema()
    assert isinstance(dct, dict)

    assert "$schema" in dct
