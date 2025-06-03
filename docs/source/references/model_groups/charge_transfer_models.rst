.. _charge_transfer:

============================
Charge Transfer models (CCD)
============================

.. important::
    This model group is only for :term:`CCD` detectors!

.. currentmodule:: pyxel.models.charge_transfer

Charge transfer models are used to manipulate data in :py:class:`~pyxel.data_structure.Pixel` array
inside the :py:class:`~pyxel.detectors.Detector` object.
Multiple models can be linked together one after another.


.. _charge_transfer_create_store_detector:

Create and Store a detector
===========================

The models :ref:`charge_transfer_save_detector` and :ref:`charge_transfer_load_detector`
can be used respectively to create and to store a :py:class:`~pyxel.detectors.Detector` to/from a file.

These models can be used when you want to store or to inject a :py:class:`~pyxel.detectors.Detector`
into the current :ref:`pipeline`.

.. _charge_transfer_save_detector:

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


.. _charge_transfer_load_detector:

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


.. _Charge Distortion Model (CDM):

Charge Distortion Model (CDM)
=============================

.. note:: This model is specific for the :term:`CCD` detector.

:guilabel:`Pixel` → :guilabel:`Pixel`

The Charge Distortion Model - CDM :cite:p:`2013:short` describes the effects of the radiation
damage causing charge deferral and image shape distortion. The analytical
model is physically realistic, yet fast enough. It was developed specifically
for the Gaia CCD operating mode, implemented in Fortran and Python. However,
a generalized version has already been applied in a broader context, for
example to investigate the impact of radiation damage on the Euclid mission.
This generalized version has been included and used in Pyxel.

Use this model to add radiation induced :term:`CTI` effects to :py:class:`~pyxel.data_structure.Pixel` array of the
to :py:class:`~pyxel.detectors.CCD` detector. Argument ``direction`` should be set as either ``"parallel"``
for parallel direction :term:`CTI` or ``"serial"`` for serial register :term:`CTI`.
User should also set arguments ``trap_release_times``, ``trap_densities`` and ``sigma``
as lists for an arbitrary number of trap species. See below for descriptions.
Other arguments include ``max_electron_volume``, ``transfer_period``,
``charge injection`` for parallel mode and ``full_well_capacity`` to override the one set in
detector :py:class:`~pyxel.detectors.Characteristics`.

.. figure:: _static/cdm.png
    :scale: 50%
    :alt: Poppy
    :align: center

    CDM (Charge Distortion Model)

.. note::
    You can find examples of this model in these Jupyter Notebooks from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_:

    * :external+pyxel_data:doc:`exposure`
    * :external+pyxel_data:doc:`examples/observation/product`


Example of the configuration file.

.. code-block:: yaml

    - name: cdm
      func: pyxel.models.charge_transfer.cdm
      enabled: true
      arguments:
        direction: "parallel"
        trap_release_times: [0.1, 1.]
        trap_densities: [0.307, 0.175]
        sigma: [1.e-15, 1.e-15]
        beta: 0.3
        max_electron_volume: 1.e-10
        transfer_period: 1.e-4
        charge_injection: true  # only used for parallel mode
        full_well_capacity: 1000.  # optional (otherwise one from detector characteristics is used)

.. autofunction:: cdm

.. _Add CTI trails (ArCTIc):

Add CTI trails
==============

:guilabel:`Pixel` → :guilabel:`Pixel`

Add image trails due to charge transfer inefficiency in :term:`CCD` detectors by modelling the
trapping, releasing, and moving of charge along pixels.

The primary inputs are the initial image followed by the properties of the :term:`CCD`,
readout electronics and trap species for serial clocking.

More information about adding :term:`CTI` trailing is described
in section 2.1 in :cite:p:`2010:massey`.


Example of the configuration file:

.. code-block:: yaml

    - name: arctic_add
      func: pyxel.models.charge_transfer.arctic_add
      enabled: true
      arguments:
        well_fill_power: 10.
        trap_densities: [1., 2., 3.]                # Add three traps
        trap_release_timescales: [10., 20., 30.]
        express: 0


.. autofunction:: arctic_add

.. _Remove CTI trails (ArCTIc):

Remove CTI trails
=================

:guilabel:`Pixel` → :guilabel:`Pixel`

Remove :term:`CTI` trails is done by iteratively modelling the addition of :term:`CTI`, as described
in :cite:p:`2010:massey` section 3.2 and Table 1.

Example of the configuration file:

.. code-block:: yaml

    - name: arctic_remove
      func: pyxel.models.charge_transfer.arctic_remove
      enabled: true
      arguments:
        well_fill_power: 10.
        instant_traps:                      # Add two traps
          - density: 1.0
            release_timescale: 10.0
          - density: 2.0
            release_timescale: 20.0
        express: 0


.. autofunction:: arctic_remove

.. _EMCCD Model:

EMCCD Model
===========

.. note:: This model is specific for the :term:`CCD` detector.

:guilabel:`Pixel` → :guilabel:`Pixel`

The Electron Multiplying CCD (EMCCD) model for the :term:`CCD` detector includes a ``multiplication_register``.
This register takes each pixel, and applies a Poisson distribution, centered around the ``total_gain``.
Each pixel is inputted and iterated through the number of ``gain_elements`` with probability of multiplication :math:`P`:

:math:`P = {G}^(\frac{1}{N_E}) - 1`

:math:`G` is the total gain, and :math:`N_E` is the number of gain elements.

The output is a :py:class:`~pyxel.data_structure.Pixel` array, with
each pixel having gone through a multiplication register.

Example of the configuration file:

.. code-block:: yaml

    - name: multiplication_register
      func: pyxel.models.charge_transfer.multiplication_register
      enabled: true
      arguments:
        gain_elements: 100
        total_gain: 1000

.. autofunction:: multiplication_register


.. _EMCCD Clock Induced Charge (CIC):

EMCCD Clock Induced Charge (CIC)
================================

:guilabel:`Pixel` → :guilabel:`Pixel`

Clock Induced Charge (CIC), can be included with ``multiplication_register_cic``.
Here a parallel CIC rate, ``pcic_rate``, and serial CIC rate ``scic_rate`` are specified,
and added to the :py:class:`~pyxel.data_structure.Pixel` array.
Each ``gain_elements`` has possibility to introduce a serial CIC event.
Serial and parallel CIC is assumed to be Poisson distributed.


Example of the configuration file:

.. code-block:: yaml

    - name: multiplication_register_cic
      func: pyxel.models.charge_transfer.multiplication_register_cic
      enabled: true
      arguments:
        gain_elements: 100
        total_gain: 1000
        pcic_rate: 0.01
        scic_rate: 0.005


.. note::
    You can find an example of this model used in this Jupyter Notebook
    :external+pyxel_data:doc:`examples/models/multiplication_register/emccd_obs`
    from `Pyxel Data <https://esa.gitlab.io/pyxel-data>`_.

.. autofunction:: multiplication_register_cic

.. note:: This model is specific for photon counting, and should be used with very low individual pixel values.