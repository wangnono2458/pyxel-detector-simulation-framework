.. _CMOS architecture:

====
CMOS
====
API reference: :py:class:`~pyxel.detectors.CMOS`

A CMOS image sensor, providing a flexible alternative to CCD detectors, functions by converting light into electrical
signals through individual pixel circuits arranged in an array. This parallel architecture facilitates faster readout
speeds and lower power consumption, leading to widespread adoption in digital cameras, smartphones, and other imaging devices.

Furthermore, CMOS technology enables the integration of additional functionalities such as on-chip analog-to-digital
conversion and image processing, enhancing versatility across various imaging applications.
However, CMOS sensors often encounter electrical crosstalk between pixels.
This interpixel capacitance (IPC) responsible for crosstalk can impact the point-spread function (PSF) of the telescope,
causing an increase in size and alteration of shape for all objects in the images while correlating with Poisson noise :cite:p:`Kannawadi_2016`.

Below all available models for CMOS are listed.

.. _CMOS models:

Available models
----------------

* Scene generation
    * :ref:`scene_generation_create_store_detector`
    * :ref:`load_star_map`
* Photon collection
    * :ref:`photon_collection_create_store_detector`
    * :ref:`simple_collection`
    * :ref:`Load image`
    * :ref:`Usaf illumination`
    * :ref:`Simple illumination`
    * :ref:`Stripe pattern`
    * :ref:`Shot noise`
    * :ref:`Physical Optics Propagation in PYthon (POPPY)`
    * :ref:`Load PSF`
    * :ref:`Wavelength dependence AIRS`
* Charge generation
    * :ref:`charge_generation_create_store_detector`
    * :ref:`Simple photoconversion`
    * :ref:`Conversion with custom QE map`
    * :ref:`Conversion with 3D QE map`
    * :ref:`Apply QE curve`
    * :ref:`Load charge`
    * :ref:`CosmiX cosmic ray model`
    * :ref:`Dark current rule07`
    * :ref:`Dark current`
    * :ref:`Simple dark current`
    * :ref:`Dark current induced`
    * :ref:`Avalanche gain`
* Charge collection
    * :ref:`charge_collection_create_store_detector`
    * :ref:`Simple collection`
    * :ref:`Simple full well`
    * :ref:`Fixed pattern noise`
    * :ref:`Inter pixel capacitance`
    * :ref:`Simple persistence`
    * :ref:`Persistence`
* Charge measurement:
    * :ref:`charge_measurement_create_store_detector`
    * :ref:`DC offset`
    * :ref:`kTC reset noise`
    * :ref:`Simple charge measurement`
    * :ref:`Output node noise CMOS`
    * :ref:`Non-linearity (polynomial)`
    * :ref:`Simple physical non-linearity`
    * :ref:`Physical non-linearity`
    * :ref:`Physical non-linearity with saturation`
    * :ref:`HxRG noise generator`
* Readout electronics:
    * :ref:`readout_electronics_create_store_detector`
    * :ref:`Simple ADC`
    * :ref:`Simple amplification`
    * :ref:`DC crosstalk`
    * :ref:`AC crosstalk`
    * :ref:`SAR ADC`
* Data processing:
    * :ref:`data_processing_create_store_detector`
    * :ref:`statistics`
    * :ref:`mean_variance`
    * :ref:`linear_regression`
    * :ref:`source_extractor`
    * :ref:`remove_cosmic_rays`
    * :ref:`snr`
