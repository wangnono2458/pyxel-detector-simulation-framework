#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Sub-package to retrieve information from JSON Schema and metadata from models."""

import json
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import pyxel.static


@cache
def get_schema() -> Mapping:
    """Retrieve the Pyxel JSON Schema.

    This function reads and parses file 'pyxel/static/pyxel_schema.json' into a dictionary.

    Returns
    -------
    dict
        Pyxel JSON Schema
    """
    # Locate the 'pyxel_schema.json' file
    folder = Path(pyxel.static.__path__[0])
    pyxel_schema_filename = folder / "pyxel_schema.json"

    # Read and parse the JSON schema
    schema = json.loads(pyxel_schema_filename.read_text())

    return schema


def clean_text(data: str) -> str:
    """Remove 'reStructuredText' Sphinx-style references.

    Parameters
    ----------
    data : str
        Text containing potential reStructuredText term references.

    Returns
    -------
    str
        Cleaned text

    Examples
    --------
    >>> clean_text("Geometrical attributes of a :term:`CCD` detector.")
    'Geometrical attributes of a CCD detector.'
    """
    return re.sub(r":term:`([^`]+)`", r"\1", data)
