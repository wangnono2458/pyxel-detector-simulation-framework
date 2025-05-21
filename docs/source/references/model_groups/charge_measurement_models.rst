.. _charge_measurement:

=========================
Charge Measurement models
=========================

.. currentmodule:: pyxel.models.charge_measurement

Charge measurement models are used to add to and manipulate data in :py:class:`~pyxel.data_structure.Signal` array
inside the :py:class:`~pyxel.detectors.Detector` object.
The values in the :py:class:`~pyxel.data_structure.Signal` array represent the amount of signal in Volt.
A charge measurement model, e.g. :ref:`Simple charge measurement`, is necessary to first convert from pixel data stored
in :py:class:`~pyxel.data_structure.Pixel` class to signal stored in :py:class:`~pyxel.data_structure.Signal`.
Multiple models are available to add detector effects after.


.. _charge_measurement_create_store_detector:

Create and Store a detector
===========================

The models :ref:`charge_measurement_save_detector` and :ref:`charge_measurement_load_detector`
can be used respectively to create and to store a :py:class:`~pyxel.detectors.Detector` to/from a file.

These models can be used when you want to store or to inject a :py:class:`~pyxel.detectors.Detector`
into the current :ref:`pipeline`.

.. _charge_measurement_save_detector:

Save detector
-------------

This model saves the current :py:class:`~pyxel.detectors.Detector` into a file.
Accepted file formats are ``.h5``, ``.hdf5``, ``.hdf`` and ``.asdf``.

.. code-block:: yaml

    - name: save_detector
      func: pyxel.models.save_detector
      enabled: true
      arguments:
        filename: my_detector.h5

.. autofunction:: pyxel.models.save_detector
   :noindex:


.. _charge_measurement_load_detector:

Load detector
-------------

This model loads a :py:class:`~pyxel.detectors.Detector` from a file and injects it in the current pipeline.
Accepted file formats are ``.h5``, ``.hdf5``, ``.hdf`` and ``.asdf``.

.. code-block:: yaml

    - name: load_detector
      func: pyxel.models.load_detector
      enabled: true
      arguments:
        filename: my_detector.h5

.. autofunction:: pyxel.models.load_detector
   :noindex:

.. _Simple charge measurement:

Simple charge measurement
=========================

:guilabel:`Pixel` → :guilabel:`Signal`

Convert the pixels array to the signal array.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_measurement
      func: pyxel.models.charge_measurement.simple_measurement
      enabled: true
      arguments:
        noise:
          - gain: 1.    # Optional

.. autofunction:: simple_measurement


.. _DC offset:

DC offset
=========

:guilabel:`Signal` → :guilabel:`Signal`

Add a DC offset to signal array of detector.

.. code-block:: yaml

    - name: dc_offset
      func: pyxel.models.charge_measurement.dc_offset
      enabled: true
      arguments:
        offset: 0.1

.. autofunction:: dc_offset

.. _Output pixel reset voltage APD:

Output pixel reset voltage APD
==============================

.. note:: This model is specific to the :term:`APD` detector.

:guilabel:`Signal` → :guilabel:`Signal`

Add output pixel reset voltage to the signal array of the :term:`APD` detector.

.. code-block:: yaml

    - name: output_pixel_reset_voltage_apd
      func: pyxel.models.charge_measurement.output_pixel_reset_voltage_apd
      enabled: true
      arguments:
        roic_drop: 3.3

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`use_cases/APD/saphira`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: output_pixel_reset_voltage_apd

.. _kTC reset noise:

kTC reset noise
===============

:guilabel:`Signal` → :guilabel:`Signal`

Add kTC reset noise to the signal array of the detector object.

.. code-block:: yaml

    - name: ktc_noise
      func: pyxel.models.charge_measurement.ktc_noise
      enabled: true
      arguments:
        node_capacitance: 30.e-15

.. note:: When using with the :term:`APD` detector, node capacitance is calculated from detector characteristics.

.. note::
    You can find examples of this model in these Jupyter Notebooks from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_:

    * :external+pyxel_data:doc:`use_cases/CMOS/cmos`
    * :external+pyxel_data:doc:`use_cases/APD/saphira`

.. autofunction:: ktc_noise

.. _Output node noise:

Output node noise
=================

.. note:: This model is specific to the :term:`CCD` detector.

:guilabel:`Signal` → :guilabel:`Signal`

Add noise to signal array of detector output node using normal random distribution.

.. code-block:: yaml

    - name: output_noise
      func: pyxel.models.charge_measurement.output_node_noise
      enabled: true
      arguments:
        std_deviation: 1.0

.. autofunction:: output_node_noise

.. _Output node noise CMOS:

Output node noise CMOS
======================

.. note:: This model is specific to the :term:`CMOS` detector.

:guilabel:`Signal` → :guilabel:`Signal`

Output node noise model for :term:`CMOS` detectors where readout is statistically independent for each pixel.

.. code-block:: yaml

    - name: output_noise
      func: pyxel.models.charge_measurement.output_node_noise_cmos
      enabled: true
      arguments:
        readout_noise: 1.0
        readout_noise_std: 2.0

.. autofunction:: output_node_noise_cmos

.. _Readout noise Saphira:

Readout noise Saphira
=====================

.. note:: This model is specific to the :term:`APD` detector.

:guilabel:`Signal` → :guilabel:`Signal`

Empirical noise for adding noise to the signal array of the :term:`APD` detector using normal random distribution.
Additional noise factor for `roic_readout_noise` is computed from detector characteristic `avalanche gain` in the model.
Noise factor based on a figure from :cite:p:`2015:rauscher` for temperature of 90K.

.. code-block:: yaml

    - name: readout_noise_saphira
      func: pyxel.models.charge_measurement.readout_noise_saphira
      enabled: true
      arguments:
        roic_readout_noise: 0.15
        controller_noise: 0.1, optional

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`use_cases/APD/saphira`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: readout_noise_saphira

.. _Non-linearity (polynomial):

Non-linearity (polynomial)
==========================

:guilabel:`Signal` → :guilabel:`Signal`

With this model you can add non-linearity to :py:class:`~pyxel.data_structure.Signal` array
to simulate the non-linearity of the output node circuit.
The non-linearity is simulated by a polynomial function.
The user specifies the polynomial coefficients with the argument ``coefficients``:
a list of :math:`n` floats e.g. :math:`[a,b,c] \rightarrow S = a + bx+ cx2` (:math:`x` is signal).

Example of the configuration file where a 10% non-linearity is introduced as a function of the signal square:

.. code-block:: yaml

    - name: linearity
      func: pyxel.models.charge_measurement.output_node_linearity_poly
      enabled: true
      arguments:
        coefficients: [0, 1, 0.9]  # e- [a,b,c] -> S = a + bx+ cx2 (x is signal)

.. note::
    You can find examples of this model in these Jupyter Notebooks from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_:

    * :external+pyxel_data:doc:`use_cases/CCD/euclid_prnu`
    * :external+pyxel_data:doc:`use_cases/HxRG/h2rg`
    * :external+pyxel_data:doc:`workshops/leiden_university_workshop/ptc`

.. autofunction:: pyxel.models.charge_measurement.output_node_linearity_poly

.. _Simple physical non-linearity:

Simple physical non-linearity
=============================

.. note:: This model is specific to the :term:`CMOS` detector.

:guilabel:`Signal` → :guilabel:`Signal`

With this model you can add non-linearity to :py:class:`~pyxel.data_structure.Signal` array.

The model assumes a planar geometry of the diode and
follows the description of the classical non-linearity model described in :cite:p:`Plazas_2017`.
It does not take into account the additional fixed capacitance and the gain non-linearity
and does not simulate saturation.

Example of the configuration file:

.. code-block:: yaml

    - name: simple_physical_non_linearity
      func: pyxel.models.charge_measurement.simple_physical_non_linearity
      enabled: true
      arguments:
        cutoff: 2.1
        n_acceptor: 1.e+18
        n_donor: 3.e+15
        diode_diameter: 10.
        v_bias: 0.1

.. autofunction:: pyxel.models.charge_measurement.simple_physical_non_linearity

.. _Physical non-linearity:

Physical non-linearity
======================

.. note:: This model is specific to the :term:`CMOS` detector.

:guilabel:`Signal` → :guilabel:`Signal`

With this model you can add non-linearity to :py:class:`~pyxel.data_structure.Signal` array.

In this simplified analytical detector non-linearity model :cite:p:`pichon`
which assumes the detector is working far from saturation,
the current flowing in the diode is restricted to a photonic current:

:math:`\frac{dV}{dt}=\frac{-I_{ph}}{C}`.

The integrating capacitance can be written as a sum of fixed capacitance :math:`C_f` in the readout integrated circuit
and diode capacitance:

:math:`C=C_f+\frac{C_0}{1-\frac{V}{V_{bi}}}`,

The diode capacitance at 0 bias :math:`C_0` is :cite:p:`sze`:

:math:`C_0=A\sqrt{\frac{e\epsilon\epsilon_0}{2V_{bi}}(\frac{1}{N_a}+\frac{1}{N_d})}`.

:math:`A` is the area of the circular shaped diode, :math:`e` electron charge,
:math:`\epsilon` dielectric constant of the material,
and :math:`N_a` and :math:`N_d` are acceptor and donor concentrations.
:math:`V_{bi}` is the built-in diode potential and is a function of :math:`N_a`, :math:`N_d`,
temperature and intrinsic carrier concentration.
By inserting second equation into first equation and integrating,
one can express voltage on the detector after exposure as a solution of a quadratic equation.
Different to the non-linearity model presented in Plazas et al. (2017) :cite:p:`Plazas_2017`,
this model takes into account the additional fixed capacitance and the gain non-linearity.
Still it is not a complete physical model and does not simulate the saturation of the detector.
Additionally, it assumes a planar geometry of the diode instead of a cylindrical.

Example of the configuration file:

.. code-block:: yaml

    - name: physical_non_linearity
      func: pyxel.models.charge_measurement.physical_non_linearity
      enabled: true
      arguments:
        cutoff: 2.1
        n_acceptor: 1.e+18
        n_donor: 3.e+15
        diode_diameter: 10.
        v_bias: 0.1
        fixed_capacitance: 5.e-15

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`examples/models/non_linearity/non_linearity`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: pyxel.models.charge_measurement.physical_non_linearity

.. _Physical non-linearity with saturation:

Physical non-linearity with saturation
======================================

.. note:: This model is specific to the :term:`CMOS` detector.

:guilabel:`Signal` → :guilabel:`Signal`

With this model you can add non-linearity to :py:class:`~pyxel.data_structure.Signal` array.

This model follows the description in :cite:p:`pichon` and gives the diode bias as a function of the time
by solving the following differential equation using the Euler method:

:math:`\frac{dV}{dt}=\frac{I(t)}{C(V)}`,

where :math:`V` is the applied bias, :math:`C` the node capacitance, and

:math:`I(t)=I_{sat}(exp(\frac{V}{nV_T})-1)-I_{ph}`.

:math:`I_{sat}` is the saturation current of the diode, :math:`V_T` is the thermal velocity,
:math:`n` is the ideality factor of the diode, and :math:`I_{ph}` is the photonic current.

Considering a diode with a cylindrical geometry, one can write

:math:`C(V)=C_f+\frac{a_0}{W_{dep}(V)}+b_0+c_0W_{dep}(V)`.

For an abrupt junction and diode dimension parameters :math:`\Phi_{imp}` and :math:`d_{imp}`,
following equations hold:

:math:`W_{dep}(V)=W_0(1-\frac{V}{V_{bi}})^{-\frac{1}{2}}`

:math:`a_0=(\frac{\Phi^2_{imp}}{4}+\Phi_{imp}d_{imp})\epsilon\pi`,

:math:`b_0=(\Phi_{imp}+d_{imp})2\epsilon\pi`,

:math:`c_0=3\epsilon\pi`.

User also specifies detector parameters ``v_reset`` and ``d_sub``.
This model assumes additional fixed capacitances,
the gain non-linearity is taken into account, it simulates the detector saturation and
assumes that the non-linearities observed mainly come from the PN junction diode.

Example of the configuration file:

.. code-block:: yaml

    - name: physical_non_linearity_with_saturation
      func: pyxel.models.charge_measurement.physical_non_linearity_with_saturation
      enabled: true
      arguments:
        cutoff: 2.1
        n_donor: 3.e+15
        n_acceptor: 1.e+18
        phi_implant: 6.e-6
        d_implant: 1.e-6
        saturation_current: 0.002
        ideality_factor: 1.34
        v_reset: 0.
        d_sub: 0.220
        fixed_capacitance: 5.e-15
        euler_points: 100

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`examples/models/non_linearity/non_linearity`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: pyxel.models.charge_measurement.physical_non_linearity_with_saturation

.. _HxRG noise generator:

HxRG noise generator
====================

:guilabel:`Pixel` → :guilabel:`Pixel`

With this model you can add noise to :py:class:`~pyxel.data_structure.Pixel` array,
before converting to :py:class:`~pyxel.data_structure.Signal` array in the charge measurement part of the pipeline.

It is a near-infrared :term:`CMOS` noise generator (ngHxRG) developed for the
James Webb Space Telescope (JWST) Near Infrared Spectrograph (NIRSpec)
described in :cite:p:`2015:rauscher`. It simulates many important noise
components including white read noise, residual bias drifts, pink 1/f
noise, alternating column noise and picture frame noise.

The model reproduces most of the Fourier noise
power spectrum seen in real data, and includes uncorrelated, correlated,
stationary and non-stationary noise components.
The model can simulate noise for HxRG detectors of
Teledyne Imaging Sensors with and without the SIDECAR ASIC IR array
controller.

* Developed by: Bernard J. Rauscher, NASA
* Developed for: James Webb Space Telescope
* Site: https://jwst.nasa.gov/publications.html


.. figure:: _static/nghxrg.png
    :scale: 50%
    :alt: nghxrg
    :align: center

    ngHxRG Noise Generator


Example of the configuration file:

.. code-block:: yaml

    - name: nghxrg
      func: pyxel.models.charge_measurement.nghxrg
      enabled: true
      arguments:
        noise:
          - ktc_bias_noise:
              ktc_noise: 1
              bias_offset: 2
              bias_amp: 2
          - white_read_noise:
              rd_noise: 1
              ref_pixel_noise_ratio: 2
          - corr_pink_noise:
              c_pink: 1.
          - uncorr_pink_noise:
              u_pink: 1.
          - acn_noise:
              acn: 1.
          - pca_zero_noise:
              pca0_amp: 1.
        window_position: [0, 0]   # Optional
        window_size: [100, 100]   # Optional
        n_output: 1
        n_row_overhead: 0
        n_frame_overhead: 0
        reverse_scan_direction: False
        reference_pixel_border_width: 4

.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`use_cases/HxRG/h2rg`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: pyxel.models.charge_measurement.nghxrg

* **kTC bias noise**
* **White readout noise**
* **Alternating column noise (ACN)**
* **Uncorrelated pink noise**
* **Correlated pink noise**
* **PCA0 noise**

..
    .. _signal_transfer:

    Signal Transfer models (CMOS)
    =============================

    .. important::
       This model group is only for :term:`CMOS`-based detectors!

    .. currentmodule:: pyxel.models.signal_transfer
    .. automodule:: pyxel.models.signal_transfer
        :members:
        :undoc-members:
        :imported-members:
        
.. _DC crosstalk:

DC crosstalk
============

:guilabel:`Signal` → :guilabel:`Signal`

Apply DC crosstalk signal to detector signal.

Example of the configuration file:

.. code-block:: yaml

    - name: dc_crosstalk
      func: pyxel.models.charge_measurement.dc_crosstalk
      enabled: true
      arguments:
        coupling_matrix: [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]]
        channel_matrix: [1,2,3,4]
        readout_directions: [1,2,1,2]

.. autofunction:: dc_crosstalk

.. _AC crosstalk:

AC crosstalk
============

:guilabel:`Signal` → :guilabel:`Signal`

Apply AC crosstalk signal to detector signal.

Example of the configuration file:

.. code-block:: yaml

    - name: ac_crosstalk
      func: pyxel.models.charge_measurement.ac_crosstalk
      enabled: true
      arguments:
        coupling_matrix: [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]]
        channel_matrix: [1,2,3,4]
        readout_directions: [1,2,1,2]

.. note::
    You can find examples of this model in these Jupyter Notebooks from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_:

    * :external+pyxel_data:doc:`examples/models/amplifier_crosstalk/crosstalk`
    * :external+pyxel_data:doc:`use_cases/HxRG/h2rg`
    * :external+pyxel_data:doc:`examples/observation/sequential`

.. autofunction:: ac_crosstalk
