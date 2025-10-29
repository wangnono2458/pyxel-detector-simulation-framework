#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from pyxel.detectors import (
    CCD,
    CCDGeometry,
    Characteristics,
    ChargeToVoltSettings,
    Environment,
    ReadoutProperties,
)
from pyxel.models.photon_collection import usaf_illumination


def test_usaf_illumination():
    """Test model 'usaf_illuminuation'."""
    # Create a fake detector
    detector = CCD(
        geometry=CCDGeometry(
            row=10,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(
            quantum_efficiency=1.0,
            charge_to_volt=ChargeToVoltSettings(value=1e-6),
            pre_amplification=1.0,
            adc_bit_resolution=16,
            adc_voltage_range=(0.0, 10.0),
        ),
    )
    detector._readout_properties = ReadoutProperties(times=[1.0])

    # Run the model
    usaf_illumination(detector=detector)
