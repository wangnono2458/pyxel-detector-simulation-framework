#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
import xarray as xr
from astropy import constants
from astropy.table import Table
from astropy.tests.helper import assert_quantity_allclose
from astropy.units import Quantity, Unit, spectral_density

from pyxel.detectors import CCD
from pyxel.models.scene_generation import load_star_map
from pyxel.models.scene_generation.load_star_map import (
    HIPPARCOS_ID,
    TYCHO2_ID,
    get_vega_spectrum_flux,
    retrieve_from_gaia,
    retrieve_from_vizier_catalog,
)
from pyxel.util import get_cache

# This is equivalent to 'import pytest_mock'
pytest_mock = pytest.importorskip(
    "pytest_mock",
    reason="Package 'pytest_mock' is not installed. Use 'pip install pytest-mock'",
)


@pytest.fixture
def source_ids() -> list[int]:
    """Return unique source identifier from the GAIA database."""
    return [66727234683960320, 65214031805717376, 65225851555715328, 65226195153096192]


@pytest.fixture
def positions(source_ids: list[int]) -> xr.Dataset:
    """Return source objects as a Dataset."""
    source_id = xr.DataArray(
        source_ids,
        dims="source_id",
    )

    ds = xr.Dataset(coords={"source_id": source_id})
    ds["ra"] = xr.DataArray(
        [56.760485086776846, 56.74561610052, 56.726951308177455, 56.736700233543914],
        dims="source_id",
        attrs={"units": "deg"},
    )

    ds["dec"] = xr.DataArray(
        [24.149991010998, 24.089174782613, 24.111718134110, 24.149504345515],
        dims="source_id",
        attrs={"units": "deg"},
    )
    ds["has_xp_sampled"] = xr.DataArray([True] * 4, dims="source_id")
    ds["phot_bp_mean_mag"] = xr.DataArray(
        [14.734505, 12.338661, 14.627676, 14.272486],
        dims="source_id",
        attrs={"units": "mag"},
    )
    ds["phot_g_mean_mag"] = xr.DataArray(
        [14.433147, 11.940813, 13.91212, 13.613182],
        dims="source_id",
        attrs={"units": "mag"},
    )
    ds["phot_rp_mean_mag"] = xr.DataArray(
        [13.954827, 11.368548, 13.091035, 12.804853],
        dims="source_id",
        attrs={"units": "mag"},
    )

    return ds


@pytest.fixture
def positions_table(positions: xr.Dataset) -> Table:
    """Return source objects as an Astropy Table."""
    table = Table.from_pandas(positions.to_pandas().reset_index())
    table["ra"].unit = "deg"
    table["dec"].unit = "deg"
    table["phot_bp_mean_mag"].unit = "mag"
    table["phot_g_mean_mag"].unit = "mag"
    table["phot_rp_mean_mag"].unit = "mag"

    return table


@pytest.fixture
def wavelengths() -> xr.DataArray:
    """Return wavelengths."""
    return xr.DataArray(
        [336.0, 338.0, 1018.0, 1020.0],
        dims="wavelength",
        attrs={"dims": "nm"},
    )


@pytest.fixture
def spectra1(wavelengths: xr.DataArray) -> xr.Dataset:
    """Return spectra for the first object."""
    ds = xr.Dataset(coords={"wavelength": wavelengths})
    ds["flux"] = xr.DataArray(
        [4.1858373e-17, 4.1012171e-17, 1.3888942e-17, 1.3445790e-17],
        dims="wavelength",
        attrs={"dims": "W / (nm * m^2)"},
    )
    ds["flux_error"] = xr.DataArray(
        [5.8010027e-18, 4.3436358e-18, 4.2265315e-18, 4.1775913e-18],
        dims="wavelength",
        attrs={"dims": "W / (nm * m^2)"},
    )

    return ds


@pytest.fixture
def spectra2(wavelengths: xr.DataArray) -> xr.Dataset:
    """Return spectra for the second object."""
    ds = xr.Dataset(coords={"wavelength": wavelengths})
    ds["flux"] = xr.DataArray(
        [3.0237057e-16, 2.9785625e-16, 2.4918341e-16, 2.5573007e-16],
        dims="wavelength",
        attrs={"dims": "W / (nm * m^2)"},
    )
    ds["flux_error"] = xr.DataArray(
        [2.8953863e-17, 2.1999507e-17, 1.7578710e-17, 1.7167379e-17],
        dims="wavelength",
        attrs={"dims": "W / (nm * m^2)"},
    )

    return ds


@pytest.fixture
def spectra3(wavelengths: xr.DataArray) -> xr.Dataset:
    """Return spectra for the third object."""
    ds = xr.Dataset(coords={"wavelength": wavelengths})
    ds["flux"] = xr.DataArray(
        [2.0389929e-17, 2.2613652e-17, 6.1739404e-17, 6.5074511e-17],
        dims="wavelength",
        attrs={"dims": "W / (nm * m^2)"},
    )
    ds["flux_error"] = xr.DataArray(
        [6.6810894e-18, 5.1004213e-18, 7.3332771e-18, 7.2768021e-18],
        dims="wavelength",
        attrs={"dims": "W / (nm * m^2)"},
    )

    return ds


@pytest.fixture
def spectra4(wavelengths: xr.DataArray) -> xr.Dataset:
    """Return spectra for the fourth object."""
    ds = xr.Dataset(coords={"wavelength": wavelengths})
    ds["flux"] = xr.DataArray(
        [2.4765946e-17, 2.1556272e-17, 9.0950110e-17, 9.1888827e-17],
        dims="wavelength",
        attrs={"dims": "W / (nm * m^2)"},
    )
    ds["flux_error"] = xr.DataArray(
        [6.9233657e-18, 5.2693949e-18, 7.4365125e-18, 7.1868190e-18],
        dims="wavelength",
        attrs={"dims": "W / (nm * m^2)"},
    )

    return ds


@pytest.fixture
def spectra_dct(
    source_ids: list[int],
    spectra1: xr.Dataset,
    spectra2: xr.Dataset,
    spectra3: xr.Dataset,
    spectra4: xr.Dataset,
) -> dict[int, tuple[Table, float]]:
    """Return spectra for the four objects as dictionary."""
    dct = {}
    for source_id, spectra in zip(
        source_ids, [spectra1, spectra2, spectra3, spectra4], strict=False
    ):
        table: Table = Table.from_pandas(spectra.to_pandas().reset_index())
        table["wavelength"].unit = "nm"
        table["flux"].unit = "W / (nm * m2)"
        table["flux_error"].unit = "W / (nm * m2)"

        dct[source_id] = table, 1.0

    return dct


@pytest.fixture
def source1_gaia(
    positions: xr.Dataset,
    source_ids: list[int],
    spectra1: xr.Dataset,
    spectra2: xr.Dataset,
    spectra3: xr.Dataset,
    spectra4: xr.Dataset,
) -> xr.Dataset:
    """Build a Dataset object retrieved from the Gaia database."""
    ds = xr.merge(
        [
            positions,
            xr.concat(
                [
                    spectra.assign(source_id=source_id)
                    for source_id, spectra in zip(
                        source_ids,
                        [spectra1, spectra2, spectra3, spectra4],
                        strict=False,
                    )
                ],
                dim="source_id",
            ),
        ]
    )
    ds["weight"] = xr.DataArray([1.0, 1.0, 1.0, 1.0], dims="source_id")

    return ds


@pytest.fixture
def source1_pyxel(
    positions: xr.Dataset,
    source_ids: list[int],
    spectra1: xr.Dataset,
    spectra2: xr.Dataset,
    spectra3: xr.Dataset,
    spectra4: xr.Dataset,
) -> xr.DataTree:
    return 0


@pytest.mark.skip(reason="Fix this test !")
@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11+")
@pytest.mark.parametrize("extrapolated_spectra", [True, False])
@patch(target="astroquery.gaia.Gaia.launch_job_async")
@patch(target="astroquery.gaia.Gaia.load_data")
def test_retrieve_from_gaia(
    mock_launch_job_async: Mock,
    mock_load_data: Mock,
    extrapolated_spectra: bool,
):
    """Test function 'retrieve_from_gaia'."""

    # Mock function 'astroquery.gaia.Gaia.launch_job_async'
    class FakeJob:
        def get_results(self):
            table = Table(
                names=[
                    "source_id",
                    "ra",
                    "dec",
                    "has_xp_sampled",
                    "phot_bp_mean_mag",
                    "phot_g_mean_mag",
                    "phot_rp_mean_mag",
                ],
                dtype=[int, float, float, bool, float, float, float],
            )
            table.add_row(
                [
                    66504343062472704,
                    Quantity(57.148003186750714, unit="deg"),
                    Quantity(23.81913615534774, unit="deg"),
                    True,
                    11.489057,
                    10.664621,
                    9.781835,
                ]
            )
            table.add_row(
                [
                    66715410636984192,
                    Quantity(56.79197187528736, unit="deg"),
                    Quantity(24.131100390060347, unit="deg"),
                    False,
                    20.105223,
                    19.483833,
                    18.774532,
                ]
            )

            return table

    mock_launch_job_async.return_value = FakeJob()
    mock_load_data.return_value = ...

    ds = retrieve_from_gaia(
        right_ascension=56.75,  # This parameter is not important
        declination=24.1167,  # This parameter is not important
        fov_radius=0.5,  # This parameter is not important
        extrapolated_spectra=extrapolated_spectra,
    )

    assert list(ds) == [
        "ra",
        "dec",
        "has_xp_sampled",
        "phot_bp_mean_mag",
        "phot_g_mean_mag",
        "phot_rp_mean_mag",
        "flux",
        "flux_error",
        "weight",
    ]
    assert dict(ds.dims) == {"source_id": 2, "wavelength": 343}


@pytest.mark.skip(reason="Fix this test !")
@patch(target="astroquery.gaia.Gaia.launch_job_async")
@patch(target="astroquery.gaia.Gaia.load_data")
def test_retrieve_from_gaia_load_data_error(
    mock_launch_job_async: Mock,
    mock_load_data: Mock,
):
    """Test function 'retrieve_from_gaia' with an error."""

    # Mock function 'astroquery.gaia.Gaia.launch_job_async'
    class FakeJob:
        def get_results(self):
            table = Table(
                names=[
                    "source_id",
                    "ra",
                    "dec",
                    "has_xp_sampled",
                    "phot_bp_mean_mag",
                    "phot_g_mean_mag",
                    "phot_rp_mean_mag",
                ],
                dtype=[int, float, float, bool, float, float, float],
            )
            table.add_row(
                [
                    66504343062472704,
                    Quantity(57.148003186750714, unit="deg"),
                    Quantity(23.81913615534774, unit="deg"),
                    True,
                    11.489057,
                    10.664621,
                    9.781835,
                ]
            )
            table.add_row(
                [
                    66715410636984192,
                    Quantity(56.79197187528736, unit="deg"),
                    Quantity(24.131100390060347, unit="deg"),
                    False,
                    20.105223,
                    19.483833,
                    18.774532,
                ]
            )

            return table

    mock_launch_job_async.return_value = FakeJob()
    mock_load_data.side_effect = requests.HTTPError()

    with pytest.raises(
        ConnectionError, match=r"Error when trying to load data from the Gaia database"
    ):
        _ = retrieve_from_gaia(
            right_ascension=56.75,  # This parameter is not important
            declination=24.1167,  # This parameter is not important
            fov_radius=0.5,  # This parameter is not important
            extrapolated_spectra=True,
        )


@patch(target="astroquery.gaia.Gaia.launch_job_async")
def test_retrieve_from_gaia_launch_job_async_error(mock_launch_job_async: Mock):
    """Test function 'retrieve_from_gaia' with an error."""
    mock_launch_job_async.side_effect = requests.HTTPError()

    with pytest.raises(ConnectionError):
        _ = retrieve_from_gaia(
            right_ascension=56.75,  # This parameter is not important
            declination=24.1167,  # This parameter is not important
            fov_radius=0.5,  # This parameter is not important
            extrapolated_spectra=False,  # This parameter is not important
        )


def test_compute_flux_compare_to_astropy():
    """Test function 'compute_flux'."""
    wavelengths = Quantity([336.0, 338.0, 1018.0, 1020.0], unit="nm")
    flux = Quantity(
        [4.1858373e-17, 4.1012171e-17, 1.3888942e-17, 1.3445790e-17],
        unit="W / (nm * m2)",
    )
    expected_flux = Quantity(
        [0.00070802, 0.00069783, 0.00071177, 0.00069041],
        unit="ph / (Angstrom s cm2)",
    )

    # Compute new flux using astropy
    new_flux = [
        y.to("ph / (Angstrom s cm2)", equivalencies=spectral_density(x))
        for x, y in zip(wavelengths, flux, strict=False)
    ]

    assert_quantity_allclose(actual=new_flux, desired=expected_flux, rtol=1e-5)


def test_compute_flux_compare_to_manual_conversion():
    """Test function 'compute_flux'."""
    wavelengths = Quantity([336.0, 338.0, 1018.0, 1020.0], unit="nm")
    flux = Quantity(
        [4.1858373e-17, 4.1012171e-17, 1.3888942e-17, 1.3445790e-17],
        unit="W / (nm * m2)",
    )
    expected_flux = Quantity(
        [0.00070802, 0.00069783, 0.00071177, 0.00069041],
        unit="ph / (Angstrom s cm2)",
    )

    # Compute new flux using the Plank's equation 'E = h * v = h * (c / wavelength)'
    photon_energy = (constants.h * constants.c / wavelengths).to("J") / Unit("ph")
    new_flux = (flux / photon_energy).to("ph / (Angstrom s cm2)")

    assert_quantity_allclose(actual=new_flux, desired=expected_flux, rtol=1e-5)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires Python 3.11+")
@patch(
    target=("pyxel.models.scene_generation.load_star_map._retrieve_objects_from_gaia")
)
def test_load_star_map(
    mock_retrieve_objects_from_gaia: Mock,
    ccd_10x10: CCD,  # Fixture
    positions_table: Table,  # Fixture
    spectra_dct: dict[int, Table],  # Fixture
    wavelengths: xr.DataArray,  # Fixture
    source1_gaia,  # Fixture
    source1_pyxel: xr.DataTree,  # Fixture
):
    """Test model 'load_star_map'."""

    # Mock function 'pyxel.models.scene_generation.load_star_map.retrieve_objects_from_gaia'
    # When this function will be called (with any parameters), it will always return this tuple
    # (positions_table, spectra_dct)
    mock_retrieve_objects_from_gaia.return_value = (positions_table, spectra_dct)

    # Run model
    detector: CCD = ccd_10x10

    # Clean the cache
    cache = get_cache()
    cache.clear()

    load_star_map(
        detector=detector,
        right_ascension=0.0,  # This parameter is not important
        declination=0.0,  # This parameter is not important
        fov_radius=0.0,  # This parameter is not important
    )

    # Check outputs
    scene = detector.scene.data
    assert isinstance(scene, xr.DataTree)

    expected_x = 3600 * source1_gaia["ra"]
    expected_y = 3600 * source1_gaia["dec"]

    exp_ds = xr.Dataset()
    exp_ds["x"] = xr.DataArray(
        np.array(expected_x), dims="ref", coords={"ref": [0, 1, 2, 3]}
    )
    exp_ds["y"] = xr.DataArray(
        np.array(expected_y), dims="ref", coords={"ref": [0, 1, 2, 3]}
    )
    exp_ds["weight"] = xr.DataArray(
        [1.0, 1.0, 1.0, 1.0],
        dims="ref",
        coords={"ref": [0, 1, 2, 3]},
    )

    x = Quantity(wavelengths, unit="nm")
    flux = Quantity(source1_gaia["flux"], unit="W / (nm * m2)")

    # Compute new flux using the Plank's equation 'E = h * v = h * (c / wavelength)'
    photon_energy = (constants.h * constants.c / x).to("J") / Unit("ph")
    new_flux = (flux / photon_energy).to("ph / (nm s cm2)")

    flux_converted: xr.DataArray = xr.zeros_like(scene["/list/0"]["flux"])
    flux_converted[:] = new_flux
    exp_ds["flux"] = flux_converted

    assert "list" in scene
    assert "0" in scene["list"]
    xr.testing.assert_allclose(scene["/list/0"].to_dataset(), exp_ds)


@pytest.mark.parametrize(
    "catalog_id",
    [
        pytest.param(HIPPARCOS_ID, id="hipparcos"),
        pytest.param(TYCHO2_ID, id="tycho"),
    ],
)
def test_retrieve_from_vizier_catalog_valid(catalog_id):
    """
    Test 'retrieve_from_vizier_catalog' for known Vizier catalogs.

    This test checks whether a valid table is returned when requesting
    sources around a given sky position. If no sources are found,
    the test is skipped.
    """

    # Sky position with higher probability of containing sources
    ra = 88.8  # Approx. Betelgeuse (bright and well known)
    dec = 7.4
    radius = 2.0  # Field of view radius in degrees

    try:
        table = retrieve_from_vizier_catalog(
            catalog_id=catalog_id,
            ra=ra,
            dec=dec,
            radius=radius,
        )

        # Check that the returned object is an Astropy Table and not empty
        assert isinstance(table, Table)
        assert len(table) > 0

    except ValueError as e:
        # If no sources are found, skip the test gracefully
        pytest.skip(f"No sources found for {catalog_id.name} catalog: {e}")


def test_retrieve_from_vizier_catalog_invalid_catalog_type():
    """Test that passing a wrong catalog value raises an error."""
    with pytest.raises(ValueError, match=r"No sources found in catalog"):
        retrieve_from_vizier_catalog(
            catalog_id="xxx",  # Wrong 'catalog_id'
            ra=88.8,
            dec=7.4,
            radius=1.0,
        )


def test_retrieve_from_vizier_catalog_no_sources():
    """Test case when the region contains no sources."""
    with pytest.raises(ValueError, match="No sources found in catalog"):
        retrieve_from_vizier_catalog(
            catalog_id=HIPPARCOS_ID,
            ra=0.0,
            dec=0.0,
            radius=0.001,  # Very small FOV in empty region
        )


def test_retrieve_from_vizier_catalog_row_limit():
    """Test that row_limit is respected."""
    table = retrieve_from_vizier_catalog(
        catalog_id=TYCHO2_ID,
        ra=88.8,
        dec=7.4,
        radius=10.0,  # large radius to ensure many stars
        row_limit=5,
    )

    assert len(table) <= 5


def test_retrieve_from_vizier_catalog_column_check():
    """
    Check that required columns exist in TYCHO2 catalog results.

    This test ensures the expected coordinate and photometry columns
    (in Vizier naming style) are present.
    """
    table = retrieve_from_vizier_catalog(
        catalog_id=TYCHO2_ID,
        ra=88.8,
        dec=7.4,
        radius=2.0,
    )

    # Match actual Vizier column names for Tycho-2
    expected_columns = {"RA(ICRS)", "DE(ICRS)", "VTmag"}  # or "BTmag"

    actual_columns = set(table.columns)
    assert expected_columns.issubset(actual_columns)


def test_retrieve_from_multiple_vizier_catalogs():
    """Ensure both hipparcos and TYCHO2 return distinct results."""
    ra, dec, radius = 88.8, 7.4, 2.0

    table_hip = retrieve_from_vizier_catalog(
        catalog_id=HIPPARCOS_ID,
        ra=ra,
        dec=dec,
        radius=radius,
    )

    table_tycho = retrieve_from_vizier_catalog(
        catalog_id=TYCHO2_ID,
        ra=ra,
        dec=dec,
        radius=radius,
    )

    assert len(table_hip) > 0
    assert len(table_tycho) > 0
    assert table_hip != table_tycho


def test_retrieve_vizier_sources_within_radius():
    from astropy.coordinates import SkyCoord

    """Check that all returned sources fall within the requested search radius."""
    ra, dec, radius_deg = 88.8, 7.4, 2.0

    table = retrieve_from_vizier_catalog(
        catalog_id=TYCHO2_ID,
        ra=ra,
        dec=dec,
        radius=radius_deg,
    )

    # Create SkyCoord for center and for returned sources
    center = SkyCoord(ra=ra, dec=dec, unit="deg")
    sources = SkyCoord(ra=table["RA(ICRS)"], dec=table["DE(ICRS)"], unit="deg")

    # Compute separations
    separations = center.separation(sources)

    # Assert all sources fall within the desired angular radius
    assert all(
        separations <= Quantity(radius_deg, unit="deg")
    ), "Some sources are outside the requested search radius."


def test_retrieve_vizier_large_radius():
    """Check catalog handles wide field queries."""
    table = retrieve_from_vizier_catalog(
        catalog_id=HIPPARCOS_ID,
        ra=88.8,
        dec=7.4,
        radius=15.0,
    )

    assert isinstance(table, Table)
    assert len(table) > 50  # Adjust based on catalog density


def test_get_vega_spectrum_flux():
    """Test function 'get_vega_spectrum_flux'."""
    data = get_vega_spectrum_flux()

    assert isinstance(data, xr.DataArray)
    assert data.ndim == 1
    assert data.attrs.get("units") == "W / (nm m2)"
