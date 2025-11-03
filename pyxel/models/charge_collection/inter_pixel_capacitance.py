#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


"""Simple Inter Pixel Capacitance (IPC) model.

This model simulates capacitive coupling between adjacent pixels in a CMOS detector.
Each pixel’s charge induces a small fraction of its signal into its neighbors,
resulting in a spatial redistribution of charge.

Reference: https://iopscience.iop.org/article/10.1088/1538-3873/128/967/095001/pdf
"""

import numpy as np
from astropy.convolution import convolve_fft

from pyxel.detectors import CMOS


def ipc_kernel(
    coupling: float, diagonal_coupling: float = 0.0, anisotropic_coupling: float = 0.0
) -> np.ndarray:
    """Generate a 3x3 IPC convolution kernel.

    The kernel defines how charge is redistributed between neighboring pixels
    due to capacitive coupling. The central element represents the fraction
    of charge retained by the pixel itself, while the off-center elements
    represent charge coupled to neighboring pixels.

    Parameters
    ----------
    coupling : float
        Coupling coefficient for direct (horizontal and vertical) neighbors.
    diagonal_coupling : float
        Coupling coefficient for diagonal neighbors.
    anisotropic_coupling : float
        Difference in coupling between the x and y directions.

    Returns
    -------
    np.ndarray
        Normalized IPC kernel representing the charge redistribution pattern.

    Raises
    ------
    ValueError
        If input parameters violate stability or coupling constraints.

    Notes
    -----
    The kernel is normalized such that its sum equals 1.
    """
    if not diagonal_coupling < coupling:
        raise ValueError("Requirement diagonal_coupling <= coupling is not met.")
    if not anisotropic_coupling < coupling:
        raise ValueError("Requirement anisotropic_coupling <= coupling is not met.")
    if not 0 <= coupling + diagonal_coupling <= 0.25:
        raise ValueError("Requirement coupling + diagonal_coupling << 1 is not met.")

    kernel_2d = np.array(
        [
            [diagonal_coupling, coupling - anisotropic_coupling, diagonal_coupling],
            [
                coupling + anisotropic_coupling,
                1 - 4 * (coupling + diagonal_coupling),
                coupling + anisotropic_coupling,
            ],
            [diagonal_coupling, coupling - anisotropic_coupling, diagonal_coupling],
        ],
        dtype=float,
    )

    return kernel_2d


def compute_ipc_convolution(
    input_2d: np.ndarray,
    coupling: float,
    diagonal_coupling: float,
    anisotropic_coupling: float,
) -> np.ndarray:
    """Compute the IPC-induced signal contribution by convolution.

    This function convolves the input charge distribution with the IPC kernel
    and returns only the *additional signal* coupled from neighboring pixels,
    i.e., excluding the pixel’s own self-contribution.

    Parameters
    ----------
    input_2d : ndarray
        2D array of pixel charge or signal before IPC coupling.
    coupling : float
        Coupling coefficient for adjacent pixels.
    diagonal_coupling : float
        Coupling coefficient for diagonal neighbors.
    anisotropic_coupling : float
        Difference in coupling between x and y directions.

    Returns
    -------
    ndarray
        2D array of same shape as `input_2d`, representing only the charge
        coupled from neighboring pixels (i.e., redistributed component).
    """
    # Generate the 3x3 IPC kernel
    kernel_2d = ipc_kernel(
        coupling=coupling,
        diagonal_coupling=diagonal_coupling,
        anisotropic_coupling=anisotropic_coupling,
    )

    # Convolution, extension on the edges with the mean value
    mean = np.mean(input_2d)
    convolved_2d = convolve_fft(input_2d, kernel_2d, boundary="fill", fill_value=mean)

    # Compute only the coupled (neighbor-induced) contribution
    ipc_contribution_2d = convolved_2d - input_2d

    return ipc_contribution_2d


def simple_ipc(
    detector: "CMOS",
    coupling: float,
    diagonal_coupling: float = 0.0,
    anisotropic_coupling: float = 0.0,
) -> None:
    """Apply a simple Inter-Pixel Capacitance (IPC) model to a CMOS detector.

    Parameters
    ----------
    detector : CMOS
    coupling : float
        Coupling coefficient for direct neighbors.
    diagonal_coupling : float, optional
        Coupling coefficient for diagonal neighbors. Default is 0.0.
    anisotropic_coupling : float, optional
        Difference between horizontal and vertical coupling. Default is 0.0.

    Notes
    -----
    The IPC effect redistributes charge between adjacent pixels without
    changing the total signal.

    For more information, you can find examples here:

    * :external+pyxel_data:doc:`examples/models/inter_pixel_capacitance/ipc`
    * :external+pyxel_data:doc:`use_cases/HxRG/h2rg`
    """
    if not isinstance(detector, CMOS):
        raise TypeError("Expecting a CMOS object for detector.")

    # Compute IPC contribution due to capacitive coupling
    ipc_contribution_2d = compute_ipc_convolution(
        input_2d=detector.pixel.non_volatile.array,
        coupling=coupling,
        diagonal_coupling=diagonal_coupling,
        anisotropic_coupling=anisotropic_coupling,
    )

    # Add the coupled charge to the detector’s volatile pixel array
    detector.pixel.volatile += ipc_contribution_2d
