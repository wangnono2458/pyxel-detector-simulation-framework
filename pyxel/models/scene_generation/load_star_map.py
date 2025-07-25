#  Copyright (c) European Space Agency, 2020.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this Pyxel package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.

"""Scene generator creates Scopesim Source object."""

import logging
import sys
import time
import warnings
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Literal

import astropy.units as u
import numpy as np
import requests
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Quantity
from astroquery.vizier import Vizier
from specutils import Spectrum
from synphot import SourceSpectrum

from pyxel import util
from pyxel.detectors import Detector

if TYPE_CHECKING:
    import pandas as pd
    from astropy.io.votable import tree
    from astropy.table import Table


class VizierCatalog(Enum):
    """Supported Vizier catalogs from Vizier service."""

    hipparcos = "I/239/hip_main"
    TYCHO2 = "I/259/tyc2"

    def id(self) -> str:
        """Return the catalog ID string for Vizier query."""
        return self.value


class CatalogType(Enum):
    GAIA = "gaia"
    hipparcos = "hipparcos"
    TYCHO2 = "tycho"

    def is_vizier(self) -> bool:
        return self in {CatalogType.hipparcos, CatalogType.TYCHO2}

    def to_vizier_catalog(self) -> VizierCatalog:
        if self == CatalogType.hipparcos:
            return VizierCatalog.hipparcos
        elif self == CatalogType.TYCHO2:
            return VizierCatalog.TYCHO2
        else:
            raise ValueError(f"{self.name} is not a Vizier catalog.")


def get_vega_spectrum(
    start_wavelength: float = 336.0,
    stop_wavelength: float = 1020.0,
    step_wavelength: float = 2.0,
) -> xr.DataArray:
    """Return the spectral energy distribution spectrum of the reference Vega star.

    Parameters
    ----------
    start_wavelength : float, optional
        The starting wavelength of the spectrum in nm. Default: 336.0
    stop_wavelength : float, optional
        The ending wavelength of the spectrum in nm. Default: 1020.0
    step_wavelength : float, optional
        The sampling interval between wavelengths in nm. Default: 2.0

    Returns
    -------
    DataArray

    Examples
    --------
    >>> spectrum = get_vega_spectrum()
    <xarray.DataArray 'flux' (wavelength: 343)> Size: 3kB
    array([3.26469623e-11, 3.26203266e-11, 3.30065817e-11, 3.26932451e-11,
    ...
           6.09989571e-12, 6.11400767e-12, 6.09219194e-12])
    Coordinates:
    * wavelength  (wavelength) float64 3kB 336.0 338.0 ... 1.018e+03 1.02e+03
    Attributes:
        units:    W / (nm m2)
    """
    # Late import
    import numpy as np
    import xarray as xr
    from astropy.units import Quantity, spectral_density
    from synphot import SourceSpectrum

    # Get the wavelengths in nm
    wavelengths_1d = Quantity(
        np.arange(
            start=start_wavelength,
            stop=stop_wavelength + step_wavelength,
            step=step_wavelength,
        ),
        unit="nm",
    )

    wavelengths = xr.DataArray(
        np.array(wavelengths_1d),
        dims="wavelength",
        coords={"wavelength": np.array(wavelengths_1d)},
        attrs={"units": str(wavelengths_1d.unit)},
    )

    # Load Vega spectrum
    vega = SourceSpectrum.from_vega()

    # Convert the Vega spectrum from "PHOTLAM" (="ph / (Angstrom s cm2)") to "W / (nm m2)"
    spectrum_1d: Quantity = vega(wavelengths_1d).to(
        "W / (nm m2)",
        equivalencies=spectral_density(wavelengths_1d),
    )

    return xr.DataArray(
        np.array(spectrum_1d),
        dims="wavelength",
        coords={"wavelength": wavelengths},
        name="flux",
        attrs={"units": str(spectrum_1d.unit)},
    )


def compute_flux(wavelength: Quantity, flux: Quantity) -> Quantity:
    """Compute flux in Photon.

    Parameters
    ----------
    wavelength : Quantity
        Unit: nm
    flux : Quantity
        Unit: W / (nm * m^2)

    Returns
    -------
    Quantity
        Unit: photon / (s * cm^2 * nm)

    >>> wavelength
    <Quantity [ 336.,  338.,  340., ..., 1016., 1018., 1020.] nm>
    >>> flux
    <Quantity [1.0647195e-14, 9.9830278e-15, 9.1758579e-15, ..., 3.5089168e-15,
               3.5376517e-15, 3.6932771e-15] W / (nm m2)>
    >>> compute_flux(wavelength=wavelength, flux=flux)
    <Quantity [0.18009338, 0.18009338, 0.18009338, ..., 0.18009338, 0.18009338,
               0.18009338] ph / (nm s cm2)>
    """
    spectrum_1d = Spectrum(spectral_axis=wavelength, flux=flux)
    source_spectrum = SourceSpectrum.from_spectrum1d(spectrum_1d)

    flux_photlam: Quantity = source_spectrum(wavelength)

    return flux_photlam.to("ph / (nm s cm2)")


def _convert_spectrum(flux_1d: np.ndarray, wavelength_1d: np.ndarray) -> np.ndarray:
    """Convert a 1d flux in W / (nm m2) to ph / (nm s cm2).

    Parameters
    ----------
    flux_1d : np.ndarray
        1D numpy array
    wavelength_1d : np.ndarray
        1D numpy array

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> flux_1d = np.array(
    ...     [2.2282904e-16, 2.4315793e-16, ..., 1.5625901e-15, 1.6213707e-15],
    ...     dtype=np.float32,
    ... )
    >>> wavelength_1d = np.array(
    ...     [3.76907117e-03, 4.13740861e-03, ..., 8.00785350e-02, 8.32541239e-02]
    ... )
    >>> _convert_spectrum(flux_1d=flux_1d, wavelength_1d=wavelength_1d)
    array([3.76907117e-03, 4.13740861e-03, ..., 8.00785350e-02, 8.32541239e-02])
    """
    converted_flux: Quantity = compute_flux(
        wavelength=Quantity(wavelength_1d, unit="nm"),
        flux=Quantity(flux_1d, unit="W / (nm m2)"),
    )

    return np.asarray(converted_flux)


def convert_spectrum(flux: xr.DataArray) -> xr.DataArray:
    """Convert a DataArray flux in W / (nm m2) to ph / (s AA cm2).

    Parameters
    ----------
    flux : DataArray
        This DataArray must have at least one dimension 'wavelength'

    Returns
    -------
    DataArray

    Examples
    --------
    >>> import xarray as xr
    >>> flux = xr.DataArray(...)
    >>> flux
    <xarray.DataArray 'flux' (source_id: 345, wavelength: 343)>
    array([[2.2282904e-16, 2.4315793e-16, ..., 1.5625901e-15, 1.6213707e-15],
           [6.8100953e-17, 6.0069631e-17, ..., 3.9121338e-17, 4.0024586e-17],
           ...,
           [5.9822517e-16, 5.6280908e-16, ..., 5.2960899e-16, 5.5697906e-16],
           [1.0647195e-14, 9.9830278e-15, ..., 3.5376517e-15, 3.6932771e-15]], dtype=float32)
    Coordinates:
      * source_id   (source_id) int64 66504343062472704 ... 65296357738731008
      * wavelength  (wavelength) float64 336.0 338.0 340.0 ... 1.018e+03 1.02e+03
    Attributes:
        units:    W / (nm m2)

    >>> convert_spectrum(flux)
    <xarray.DataArray 'flux' (source_id: 345, wavelength: 343)>
    array([[3.76907117e-02, 4.13740861e-02, ..., 8.00785350e-01, 8.32541239e-01],
           [1.15190254e-02, 1.02210366e-02, ..., 2.00486326e-02, 2.05518196e-02],
           ...,
           [1.01187592e-01, 9.57637374e-02, ..., 2.71410354e-01, 2.85997559e-01],
           [1.80093381e+00, 1.69864354e+00, ..., 1.81295134e+00, 1.89642359e+00]])
    Coordinates:
      * source_id   (source_id) int64 66504343062472704 ... 65296357738731008
      * wavelength  (wavelength) float64 336.0 338.0 340.0 ... 1.018e+03 1.02e+03
    Attributes:
        units:    ph / (nm s cm2)
    """
    # Convert flux in W / (nm m2) to ph / (nm s cm2)
    new_flux: xr.DataArray = xr.apply_ufunc(
        _convert_spectrum,
        flux,
        kwargs={"wavelength_1d": flux["wavelength"].to_numpy()},
        vectorize=True,
        input_core_dims=[["wavelength"]],
        output_core_dims=[["wavelength"]],
    )

    new_flux.attrs["units"] = "ph / (nm s cm2)"
    new_flux["wavelength"].attrs["units"] = flux["wavelength"].units
    return new_flux


def _retrieve_objects_from_gaia(
    right_ascension: float,
    declination: float,
    fov_radius: float,
    extrapolated_spectra: bool,
) -> tuple["Table", dict[int, tuple["Table", float]]]:
    """Query the GAIA Catalog to retrieve sources and their spectra near given sky coordinates.

    The function performs a cone search around a specified right ascension (RA), declination (DEC),
    and field-of-view (FOV) radius using the Gaia archive.

    If `extrapolated_spectra` is `True`, sources without Gaia XP spectra will be assigned an extrapolated A0V spectrum
    scaled by their G-band magnitude.

    Columns description:
    * ``source_id``: Unique source identifier of the source
    * ``ra``: Barycentric right ascension of the source.
    * ``dec``: Barycentric Declination of the source.
    * ``has_xp_sampled``: Flag indicating the availability of mean BP/RP spectrum in sampled form for this source
    * ``phot_bp_mean_mag``: Mean magnitude in the integrated BP band.
    * ``phot_g_mean_mag``: Mean magnitude in the G band.
    * ``phot_rp_mean_mag``: Mean magnitude in the integrated RP band.

    Parameters
    ----------
    right_ascension: float
        Right ascension (RA) of the center of the search cone, in degrees.
    declination: float
        Declination (DEC) of the center of the search cone, in degrees.
    fov_radius: float
        Radius of the search cone (field-of-view), in degrees.
    extrapolated_spectra : bool
        If True, generates extrapolated A0V spectra for sources without Gaia XP spectra.
        If False, only sources with XP spectra will be included.

    Returns
    -------
    Table, Sequence of Tables
        All sources found and a list of spectrum for these sources

    Raises
    ------
    ConnectionError
        If the connection to the GAIA database cannot be established.

    Notes
    -----
    More information about the GAIA catalog at these links:
    * https://gea.esac.esa.int/archive/documentation/GDR3/
    * https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html

    Examples
    --------
    >>> positions, (spectra, weight) = _retrieve_objects_from_gaia(
    ...     right_ascension=56.75,
    ...     declination=24.1167,
    ...     fov_radius=0.05,
    ... )
    >>> positions
        source_id             ra                dec         has_xp_sampled phot_bp_mean_mag phot_g_mean_mag phot_rp_mean_mag
                             deg                deg                              mag              mag             mag
    ----------------- ------------------ ------------------ -------------- ---------------- --------------- ----------------
    66727234683960320 56.760485086776846 24.149991010998228           True        14.734505       14.433147        13.954827
    65214031805717376     56.74561610052 24.089174782613686           True        12.338661       11.940813        11.368548
    65225851555715328 56.726951308177455 24.111718134110838           True        14.627676        13.91212        13.091035
    65226195153096192 56.736700233543914 24.149504345515066           True        14.272486       13.613182        12.804853
    >>> len(spectra)
    4
    >>> spectra[66727234683960320]
    wavelength      flux       flux_error
        nm      W / (nm m2)   W / (nm m2)
    ---------- ------------- -------------
         336.0 4.1858373e-17 5.8010027e-18
         338.0  4.101217e-17  4.343636e-18
         340.0  3.499973e-17 3.4906054e-18
         342.0 3.0911544e-17 3.0178371e-18
           ...           ...           ...
        1014.0  1.742315e-17  4.146798e-18
        1016.0 1.5590336e-17  4.358966e-18
        1018.0 1.3888942e-17 4.2265315e-18
        1020.0  1.344579e-17 4.1775913e-18
    """
    # Late import
    from astropy.table import Table
    from astroquery.gaia import Gaia

    # Unlimited rows.
    Gaia.ROW_LIMIT = -1
    # we get the data from GAIA DR3
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    # Prepare the query
    query = (
        "SELECT source_id, ra, dec, has_xp_sampled, phot_bp_mean_mag, phot_g_mean_mag, phot_rp_mean_mag "
        "FROM gaiadr3.gaia_source "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec),CIRCLE('ICRS',{right_ascension},{declination},{fov_radius}))=1"
    )

    if not extrapolated_spectra:
        query += " AND has_xp_sampled = 'True'"

    try:
        # Query for the catalog to search area with coordinates in FOV of optics.
        job = Gaia.launch_job_async(query)
    except requests.HTTPError as exc:
        my_exception = ConnectionError(
            "Error when trying to retrieve sources from the Gaia database"
        )

        if sys.version_info >= (3, 11):
            my_exception.add_note(
                f"Failed to retrieve the spectra with parameters {right_ascension=}, {declination=}, {fov_radius=}"
            )
            my_exception.add_note(f"{query=}")

        raise my_exception from exc

    # get the results from the query job
    results_table: Table = job.get_results()
    df: pd.DataFrame = results_table.to_pandas()

    # set parameters to load data from Gaia catalog
    retrieval_type = "XP_SAMPLED"
    data_release = "Gaia DR3"
    data_structure = "INDIVIDUAL"

    # Get all sources
    if "SOURCE_ID" in df.columns:
        source_key = "SOURCE_ID"
    elif "source_id" in df.columns:
        # This change in 'astroquery' 0.4.8+
        source_key = "source_id"
    else:
        raise ValueError(
            f"Expecting row 'SOURCE_ID' or 'source_id' in 'result'. Got these keys: {df.columns=}"
        )

    #########################################################################################
    # Get source(s) without spectrum                                                        #
    # Columns:                                                                              #
    #   - 'phot_bp_mean_mag': integrated Blue Photometer (330 nm to 680 nm) mean magnitude  #
    #   - 'phot_g_mean_mag': integrated Gaia band (330 nm to 1050 nm) mean magnitude        #
    #   - 'phot_rp_mean_mag': integrated Red Photometer (640 nm to 1050 nm) mean magnitude  #
    #########################################################################################
    df_without_spectra: pd.DataFrame = df.query("has_xp_sampled == False")[
        [source_key, "phot_g_mean_mag"]
    ]

    # Compute weight (= flux / flux_vega)
    # The Vega magnitude for column 'phot_g_mean_mag' is computed with this formula:
    #   m = -2.5 * log10(flux / flux_vega)
    # Therefore:
    #   flux / flux_vega = 10 ** (m / -2.5) = 10 ** (-0.4 * m)
    df_without_spectra["weight"] = df_without_spectra["phot_g_mean_mag"].map(
        lambda x: 10 ** (-0.4 * x)
    )

    vega_dataarray: xr.DataArray = get_vega_spectrum(
        start_wavelength=336.0, stop_wavelength=1020.0, step_wavelength=2.0
    )

    vega_table: Table = Table.from_pandas(
        vega_dataarray.to_pandas().reset_index("wavelength"),
        units={
            "wavelength": vega_dataarray["wavelength"].units,
            "flux": vega_dataarray.units,
        },
    )

    # TODO: Build a Dataset with
    #       - weight   (ref)
    #       - flux     (ref, wavelength)
    spectra_extrapolated: dict[int, tuple[Table, float]] = {
        int(row[source_key]): (vega_table, float(row["weight"]))
        for _, row in df_without_spectra.iterrows()
    }

    #####################################
    # Get source(s) with spectrum       #
    #####################################
    # Get the unique source identifiers (unique within a particular data release)
    source_ids_with_spectra: pd.Series = df.query("has_xp_sampled == True")[source_key]
    if len(source_ids_with_spectra) > 5000:
        # TODO: Fix this
        raise NotImplementedError("Cannot retrieve more than 5000 sources")

    try:
        # load spectra from stars
        spectra_dct: dict[str, list[tree.Table]] = Gaia.load_data(
            ids=source_ids_with_spectra,
            retrieval_type=retrieval_type,
            data_release=data_release,
            data_structure=data_structure,
            format="votable",  # Note: It's not yet possible to use format 'votable_gzip'
        )
    except requests.HTTPError as exc:
        my_exception = ConnectionError(
            "Error when trying to load data from the Gaia database"
        )

        if sys.version_info >= (3, 11):
            my_exception.add_note(
                f"Failed to retrieve the spectra with parameters {retrieval_type=}, {data_release=} and {data_structure=}"
            )

        raise my_exception from exc

    # Extract and combine the spectra
    spectra: dict[int, tuple[Table, float]] = {}
    for xml_filename, all_spectra in spectra_dct.items():
        try:
            for spectrum in all_spectra:
                source_id = int(spectrum.get_field_by_id("source_id").value)
                spectra[source_id] = (spectrum.to_table(), 1.0)

        except Exception as exc:
            if sys.version_info >= (3, 11):
                exc.add_note(f"Cannot extract spectra from {xml_filename=}")

            raise

    return results_table, spectra | spectra_extrapolated


def retrieve_from_gaia(
    right_ascension: float,
    declination: float,
    fov_radius: float,
    extrapolated_spectra: bool,
) -> xr.Dataset:
    """Query the GAIA Catalog to retrieve sources and their spectra near given sky coordinates.

    The function performs a cone search around a specified right ascension (RA), declination (DEC),
    and field-of-view (FOV) radius using the Gaia archive.

    If `extrapolated_spectra` is `True`, sources without Gaia XP spectra will be assigned an extrapolated A0V spectrum
    scaled by their G-band magnitude.

    Data variable/coordinates description:
    * ``source_id``: Unique source identifier of the source
    * ``ra``: Barycentric right ascension of the source.
    * ``dec``: Barycentric Declination of the source.
    * ``has_xp_sampled``: Flag indicating the availability of mean BP/RP spectrum in sampled form for this source
    * ``phot_bp_mean_mag``: Mean magnitude in the integrated BP band.
    * ``phot_g_mean_mag``: Mean magnitude in the G band.
    * ``phot_rp_mean_mag``: Mean magnitude in the integrated RP band.

    Parameters
    ----------
    right_ascension: float
        Right ascension (RA) of the center of the search cone, in degrees.
    declination: float
        Declination (DEC) of the center of the search cone, in degrees.
    fov_radius: float
        Radius of the search cone (field-of-view), in degrees.
    extrapolated_spectra : bool
        If True, generates extrapolated A0V spectra for sources without Gaia XP spectra.
        If False, only sources with XP spectra will be included.

    Returns
    -------
    Dataset

    Raises
    ------
    ConnectionError
        If the connection to the GAIA database cannot be established.

    Notes
    -----
    More information about the GAIA catalog at these links:
    * https://gea.esac.esa.int/archive/documentation/GDR3/
    * https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html

    Examples
    --------
    >>> ds = retrieve_from_gaia(
    ...     right_ascension=56.75,
    ...     declination=24.1167,
    ...     fov_radius=0.05,
    ... )
    >>> ds
    <xarray.Dataset>
    Dimensions:           (source_id: 345, wavelength: 343)
    Coordinates:
      * source_id         (source_id) int64 66504343062472704 ... 65296357738731008
      * wavelength        (wavelength) float64 336.0 338.0 ... 1.018e+03 1.02e+03
    Data variables:
        ra                (source_id) float64 57.15 57.18 57.23 ... 56.4 56.42 56.39
        dec               (source_id) float64 23.82 23.83 23.88 ... 24.43 24.46
        has_xp_sampled    (source_id) bool True True True True ... True True True
        phot_bp_mean_mag  (source_id) float32 11.49 14.13 15.22 ... 11.51 8.727
        phot_g_mean_mag   (source_id) float32 10.66 13.8 14.56 ... 14.79 11.09 8.55
        phot_rp_mean_mag  (source_id) float32 9.782 13.29 13.78 ... 14.19 10.5 8.229
        flux              (source_id, wavelength) float32 2.228e-16 ... 3.693e-15
        flux_error        (source_id, wavelength) float32 7.737e-17 ... 2.783e-16
    """
    positions_table: Table
    spectra_dct: dict[int, tuple[Table, float]]
    positions_table, spectra_dct = _retrieve_objects_from_gaia(
        right_ascension=right_ascension,
        declination=declination,
        fov_radius=fov_radius,
        extrapolated_spectra=extrapolated_spectra,
    )

    # Convert data from Gaia into a dataset
    positions: xr.Dataset = (
        positions_table.to_pandas()
        .rename(columns={"SOURCE_ID": "source_id"})
        .set_index("source_id")
        .to_xarray()
    )
    spectra: xr.Dataset = xr.concat(
        [
            spectrum_table.to_pandas(index="wavelength")
            .to_xarray()
            .assign_coords(source_id=source_id)
            .assign(weight=weight)
            for source_id, (spectrum_table, weight) in spectra_dct.items()
        ],
        dim="source_id",
    )

    ds: xr.Dataset = xr.merge([positions, spectra])

    # Add units
    first_spectrum: Table
    first_spectrum, _ = next(iter(spectra_dct.values()))

    ds["wavelength"].attrs = {"units": str(first_spectrum["wavelength"].unit)}
    ds["flux"].attrs = {"units": str(first_spectrum["flux"].unit)}
    # ds["flux_error"].attrs = {"units": str(first_spectrum["flux_error"].unit)}
    # ds["flux_photlam"].attrs = {"units": str(first_spectrum["flux_photlam"].unit)}

    ds["ra"].attrs = {
        "name": "Right Ascension",
        "units": str(positions_table["ra"].unit),
    }
    ds["dec"].attrs = {"name": "Declination", "units": str(positions_table["dec"].unit)}
    # ds["phot_bp_mean_mag"].attrs = {
    #     "name": "Mean magnitude in the integrated BP band (from 330 nm to 680 nm)",
    #     "units": str(positions_table["phot_bp_mean_mag"].unit),
    # }
    ds["phot_g_mean_mag"].attrs = {
        "name": "Mean magnitude in the G band (from 330 nm to 1050 nm)",
        "units": str(positions_table["phot_g_mean_mag"].unit),
    }
    # ds["phot_rp_mean_mag"].attrs = {
    #     "name": "Mean magnitude in the integrated RP band (from 640 nm to 1050 nm)",
    #     "units": str(positions_table["phot_rp_mean_mag"].unit),
    # }

    return ds


def _load_objects_from_gaia(
    right_ascension: float,
    declination: float,
    fov_radius: float,
    extrapolated_spectra: bool,
) -> xr.Dataset:
    """Load objects from GAIA Catalog for given coordinates and FOV.

    Parameters
    ----------
    right_ascension: float
        Right ascension (RA) of the center of the search cone, in degrees.
    declination: float
        Declination (DEC) of the center of the search cone, in degrees.
    fov_radius: float
        Radius of the search cone (field-of-view), in degrees.
    extrapolated_spectra : bool
        If True, generates extrapolated A0V spectra for sources without Gaia XP spectra.
        If False, only sources with XP spectra will be included.

    Returns
    -------
    Dataset
        Dataset object in the FOV at given coordinates found by the GAIA catalog.

    Examples
    --------
    >>> ds = _load_objects_from_gaia(
    ...     right_ascension=56.75,
    ...     declination=24.1167,
    ...     fov_radius=0.5,
    ... )
    >>> ds
    <xarray.Dataset>
    Dimensions:     (ref: 345, wavelength: 343)
    Coordinates:
      * ref         (ref) int64 0 1 2 3 4 5 6 7 ... 337 338 339 340 341 342 343 344
      * wavelength  (wavelength) float64 336.0 338.0 340.0 ... 1.018e+03 1.02e+03
    Data variables:
        x           (ref) float64 1.334e+03 1.434e+03 ... -1.271e+03 -1.381e+03
        y           (ref) float64 -1.009e+03 -956.1 -797.1 ... 1.195e+03 1.309e+03
        weight      (ref) float64 11.49 14.13 15.22 14.56 ... 15.21 11.51 8.727
        flux        (ref, wavelength) float64 2.228e-16 2.432e-16 ... 3.693e-15
    """
    # TODO: Fix this. See issue #81
    # logging.getLogger("astroquery").setLevel(logging.WARNING)

    # Get all sources and spectrum in one Dataset
    ds_from_gaia: xr.Dataset = retrieve_from_gaia(
        right_ascension=right_ascension,
        declination=declination,
        fov_radius=fov_radius,
        extrapolated_spectra=extrapolated_spectra,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Convert 'flux' from W / (nm m2) to ph / (nm s cm2)
        flux_converted: xr.DataArray = convert_spectrum(flux=ds_from_gaia["flux"])

    # Convert the positions to arcseconds and center around 0.0, 0.0.
    # The centering is necessary because ScopeSIM cannot actually 'point' the telescope.
    ra_arcsec: Quantity = Quantity(ds_from_gaia["ra"], unit="deg").to("arcsec")
    dec_arcsec: Quantity = Quantity(ds_from_gaia["dec"], unit="deg").to("arcsec")

    x: Quantity = ra_arcsec  # - ra_arcsec.mean()
    y: Quantity = dec_arcsec  # - dec_arcsec.mean()

    num_sources = len(ds_from_gaia["source_id"])
    ref_sequence: Sequence[int] = range(num_sources)

    ds = xr.Dataset(coords={"ref": ref_sequence})
    ds["x"] = xr.DataArray(np.asarray(x), dims="ref", attrs={"units": str(x.unit)})
    ds["y"] = xr.DataArray(np.asarray(y), dims="ref", attrs={"units": str(y.unit)})
    ds["weight"] = xr.DataArray(
        np.asarray(ds_from_gaia["weight"], dtype=float),
        dims="ref",
        attrs={"units": "", "name": "weight"},
    )
    ds["flux"] = xr.DataArray(
        np.asarray(flux_converted, dtype=float),
        dims=["ref", "wavelength"],
        coords={"wavelength": flux_converted["wavelength"]},
        attrs={"units": flux_converted.attrs["units"]},
    )

    ds.attrs = {
        "right_ascension": str(Quantity(right_ascension, unit="deg")),
        "declination": str(Quantity(declination, unit="deg")),
        "fov_radius": str(Quantity(fov_radius, unit="deg")),
    }

    return ds


# TODO: add information about magnitude
# TODO: add option to select filter to compute apparent magnitude
# TODO: add option to select different catalogue versions


def load_objects_from_gaia(
    right_ascension: float,
    declination: float,
    fov_radius: float,
    extrapolated_spectra: bool,
    with_caching: bool = True,
) -> xr.Dataset:
    """Load objects from GAIA Catalog for given coordinates and FOV.

    Parameters
    ----------
    right_ascension: float
        Right ascension (RA) of the center of the search cone, in degrees.
    declination: float
        Declination (DEC) of the center of the search cone, in degrees.
    fov_radius: float
        Radius of the search cone (field-of-view), in degrees.
    extrapolated_spectra : bool
        If True, generates extrapolated A0V spectra for sources without Gaia XP spectra.
        If False, only sources with XP spectra will be included.
    with_caching : bool
        Enable/Disable caching request to GAIA catalog.

    Returns
    -------
    Dataset
        Dataset object in the FOV at given coordinates found by the GAIA catalog.

    Examples
    --------
    >>> ds = load_objects_from_gaia(
    ...     right_ascension=56.75,
    ...     declination=24.1167,
    ...     fov_radius=0.5,
    ... )
    >>> ds
    <xarray.Dataset>
    Dimensions:     (ref: 345, wavelength: 343)
    Coordinates:
      * ref         (ref) int64 0 1 2 3 4 5 6 7 ... 337 338 339 340 341 342 343 344
      * wavelength  (wavelength) float64 336.0 338.0 340.0 ... 1.018e+03 1.02e+03
    Data variables:
        x           (ref) float64 1.334e+03 1.434e+03 ... -1.271e+03 -1.381e+03
        y           (ref) float64 -1.009e+03 -956.1 -797.1 ... 1.195e+03 1.309e+03
        weight      (ref) float64 11.49 14.13 15.22 14.56 ... 15.21 11.51 8.727
        flux        (ref, wavelength) float64 2.228e-16 2.432e-16 ... 3.693e-15
    """
    # Define a unique key to find/retrieve data in the cache
    key_cache = (
        __name__,
        right_ascension,
        declination,
        fov_radius,
        extrapolated_spectra,
    )

    start_time: float = time.perf_counter()

    if with_caching and key_cache in (cache := util.get_cache()):
        # Retrieve cached dataset
        ds: xr.Dataset = cache[key_cache]

        end_time: float = time.perf_counter()
        logging.info(
            "Retrieve 'dataset' for model 'load_star_map' from cache '%r' in %f s",
            cache.directory,
            end_time - start_time,
        )

    else:
        ds = _load_objects_from_gaia(
            right_ascension=right_ascension,
            declination=declination,
            fov_radius=fov_radius,
            extrapolated_spectra=extrapolated_spectra,
        )

        if with_caching:
            # Store dataset in the cache
            cache[key_cache] = ds

            end_time = time.perf_counter()
            logging.info(
                "Store 'dataset' for model 'load_star_map' in cache '%r' in %f s",
                cache.directory,
                end_time - start_time,
            )

    return ds


def retrieve_from_vizier_catalog(
    ra: float,
    dec: float,
    radius: float,
    catalog: VizierCatalog,
    row_limit: int = -1,
) -> Table:
    """Retrieve sources from a Vizier catalog given coordinates and FOV.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    radius : float
        Search radius in degrees.
    catalog : VizierCatalog
        Catalog enum (HIPPARCOS, TYCHO2).
    row_limit : int
        Maximum number of rows to return. Use -1 for unlimited.

    Returns
    -------
    Table
        Astropy Table with source data.

    Raises
    ------
    ValueError
        If no results are found or required columns are missing.
    """
    # Initialize the VizieR query interface
    vizier = Vizier(columns=["*"])
    vizier.ROW_LIMIT = row_limit

    # Perform the region query
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    result = vizier.query_region(
        coord,
        radius=Quantity(radius, unit="deg"),
        catalog=catalog.id(),
    )

    if not result:
        raise ValueError(
            f"No sources found in catalog '{catalog.name}' at given coordinates."
        )

    table: Table = result[0]

    if len(table) == 0:
        raise ValueError(f"Catalog '{catalog.name}' returned an empty table.")

    return table


# TODO: Keep only data variables 'ra', 'dec' and 'mag' ?
def normalize_vizier_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize field names in a Vizier-derived xarray.Dataset.

    This function renames common coordinate/magnitude fields to standard names
    used across different catalogs.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from Vizier query.

    Returns
    -------
    xr.Dataset
        Dataset with normalized variable names: 'ra', 'dec', 'mag'...
    """
    rename_map = {}

    # Coordinate fields (most common variants)
    for ra_candidate in ["RAICRS", "RA_ICRS", "RAJ2000", "RAdeg"]:
        if ra_candidate in ds:
            rename_map[ra_candidate] = "ra"
            break

    for dec_candidate in ["DEICRS", "DE_ICRS", "DEJ2000", "DEdeg"]:
        if dec_candidate in ds:
            rename_map[dec_candidate] = "dec"
            break

    # Magnitude fields (often named differently)
    for mag_candidate in ["Hpmag", "VTmag", "Vmag", "BTmag", "phot_g_mean_mag"]:
        if mag_candidate in ds:
            rename_map[mag_candidate] = "mag"
            break

    # Apply renaming
    # ds = ds.rename(rename_map)

    # # Optional: units
    # if "ra" in ds:
    #     ds["ra"].attrs.setdefault("units", "deg")
    # if "dec" in ds:
    #     ds["dec"].attrs.setdefault("units", "deg")
    # if "mag" in ds:
    #     ds["mag"].attrs.setdefault("units", "mag")

    return ds.rename(rename_map)


# TODO: This could be a more general function to convert from a Table to a Dataset
def convert_vizier_table_to_dataset(table: "Table") -> xr.Dataset:
    """
    Convert an Astropy Table (from Vizier) into an xarray.Dataset.

    Parameters
    ----------
    table : Table
        Astropy Table returned by Vizier.

    Returns
    -------
    Dataset
        Dataset with all columns and metadata from the table.
    """
    # df = table.to_pandas().reset_index(drop=True)
    df = table.to_pandas().set_index("HIP")
    ds = df.to_xarray()

    # Copy units and description (if present) from the Table to Dataset attrs
    for col in table.colnames:
        if table[col].unit:
            ds[col].attrs["units"] = str(table[col].unit)

        if table[col].description:
            ds[col].attrs["long_name"] = table[col].description

    return ds


def skycoord_to_xy(
    ra: np.ndarray,
    dec: np.ndarray,
    center_ra: float,
    center_dec: float,
    fov_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    sources = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    center = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame="icrs")

    separation = center.separation(sources).deg
    angle = center.position_angle(sources).rad

    dx = separation * np.sin(angle)
    dy = separation * np.cos(angle)

    x = dx / fov_radius
    y = dy / fov_radius

    return x, y


# TODO: Rewrite doc and rename it '_load_objects_from_vizier'
def load_objects_from_vizier(
    right_ascension: float,
    declination: float,
    fov_radius: float,
    catalog: VizierCatalog,
) -> xr.Dataset:
    """Load sources from Vizier catalog and return them as a Pyxel-ready Dataset,
    with positions in arcsec centered on the field center, matching the GAIA logic.
    """
    table: Table = retrieve_from_vizier_catalog(
        ra=right_ascension,
        dec=declination,
        radius=fov_radius,
        catalog=catalog,
    )

    ds = convert_vizier_table_to_dataset(table)
    ds = normalize_vizier_dataset(ds)

    # TODO: ds['mag'] is a magnitude in Johnson V (H5)
    #       Do we have to first get the Vega spectrum and then apply this bandpass filter ?
    #       import matplotlib.pyplot as plt
    #       from synphot import SpectralElement
    #       v = SpectralElement.from_filter('johnson_v')
    #       plt.plot(v.waveset, v(v.waveset))
    # ra = ds["ra"].values
    # dec = ds["dec"].values
    mag = ds["mag"].values
    flux = 10 ** (-0.4 * mag)
    # weight = flux.copy()

    # TODO: Compute weight based on 'mag'
    # TODO: Get flux from 'Vega' ? or A0V ...?

    if catalog == VizierCatalog.TYCHO2:
        wavelength = 530.0
    elif catalog == VizierCatalog.hipparcos:
        wavelength = 518.0
    else:
        wavelength = 550.0

    # # RA/DEC offset from center → in degrees
    # x_deg, y_deg = skycoord_to_xy(
    #     ra=ra,
    #     dec=dec,
    #     center_ra=right_ascension,
    #     center_dec=declination,
    #     fov_radius=fov_radius,
    # )

    # Convert to arcsec (matching GAIA logic)
    # x_arcsec = x_deg * 3600
    # y_arcsec = y_deg * 3600

    ra = Quantity(ds["ra"], unit=ds["ra"].units).to("arcsec")
    dec = Quantity(ds["dec"], unit=ds["dec"].units).to("arcsec")

    ref = np.arange(len(ra))

    dataset = xr.Dataset(
        data_vars={
            "x": (("ref",), ra.value, {"units": str(ra.unit)}),
            "y": (("ref",), dec.value, {"units": str(dec.unit)}),
            "weight": (("ref",), np.ones_like(flux), {"units": ""}),
            "flux": (
                ("ref", "wavelength"),
                np.expand_dims(flux, axis=1),
                {"units": "ph / (s * nm * cm2)"},
            ),
        },
        coords={
            "ref": ref,
            "wavelength": np.array([wavelength]),
        },
        attrs={
            "catalog": catalog.name,
            "right_ascension": str(Quantity(right_ascension, "deg")),
            "declination": str(Quantity(declination, "deg")),
            "fov_radius": str(Quantity(fov_radius, "deg")),
        },
    )

    return dataset


def load_star_map(
    detector: Detector,
    right_ascension: float,
    declination: float,
    fov_radius: float,
    extrapolated_spectra: bool = True,
    band: Literal["Vmag", "VTmag"] | None = None,
    catalog: Literal["gaia", "tycho", "hipparcos"] = "gaia",
    with_caching: bool = True,
):
    """Generate scene from scopesim Source object loading stars from the selected catalog.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    right_ascension : float
        Right ascension (RA) of the pointing center in degree.
    declination : float
        Declination (DEC) of the pointing center in degree.
    fov_radius : float
        Radius of the field of view (FOV) around the pointing center in degree.
    extrapolated_spectra : bool, optional
        If True (default), extrapolates A0V spectra for Gaia sources that lack XP spectra.
    with_caching : bool
        Enable/Disable caching queries.

    Notes
    -----
    For more information, you can find an example here:
    :external+pyxel_data:doc:`examples/models/scene_generation/tutorial_example_scene_generation`.
    """
    cat_type = CatalogType(catalog)

    # Band validation
    valid_band_map = {CatalogType.TYCHO2: {"VTmag"}, CatalogType.hipparcos: {"Vmag"}}

    if band not in valid_band_map.get(cat_type, set()):
        raise ValueError(
            f"Invalid band '{band}' for catalog '{catalog}'. "
            f"Allowed: {valid_band_map.get(cat_type, set())}"
        )

    # Load data
    if cat_type == CatalogType.GAIA:
        ds = load_objects_from_gaia(
            right_ascension=right_ascension,
            declination=declination,
            fov_radius=fov_radius,
            extrapolated_spectra=extrapolated_spectra,
            with_caching=with_caching,
        )
    else:
        vizier_cat: VizierCatalog = cat_type.to_vizier_catalog()
        ds = load_objects_from_vizier(
            right_ascension=right_ascension,
            declination=declination,
            fov_radius=fov_radius,
            catalog=vizier_cat,
        )

    detector.scene.add_source(ds)
