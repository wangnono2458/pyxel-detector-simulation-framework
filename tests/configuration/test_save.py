#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path
from shutil import copy2

import pytest

from pyxel.configuration import copy_config_file


@pytest.fixture
def valid_filename(request: pytest.FixtureRequest) -> Path:
    """Get a valid existing YAML filename."""
    folder = Path(request.module.__file__).parent
    filename: Path = folder / "data/exposure1.yaml"
    return filename.resolve(strict=True)


@pytest.mark.parametrize(
    "filename",
    [
        "my_config.yaml",
        "other_config.yml",
        "test.YAML",
        "another.YML",
    ],
)
def test_save_with_yaml(filename, tmp_path: Path, valid_filename: Path):
    """Test function 'save'."""
    input_folder: Path = tmp_path / "input"
    input_folder.mkdir()

    # Create an input YAML file
    input_filename: Path = input_folder / filename
    copy2(src=valid_filename, dst=input_filename)
    assert input_filename.exists()

    # Create an output folder
    output_folder: Path = tmp_path / "output"
    output_folder.mkdir()

    # Test
    filename_copied = copy_config_file(
        input_filename=input_filename, output_dir=output_folder
    )
    exp_filename_copied = tmp_path / "output" / filename

    assert filename_copied.exists()
    assert filename_copied == exp_filename_copied
