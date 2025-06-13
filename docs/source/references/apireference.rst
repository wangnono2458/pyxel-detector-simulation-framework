.. _apireference:

=============
API reference
=============

This page provides an auto-generated summary of Pyxel's API.

Top-level functions
===================

.. currentmodule:: pyxel

.. autosummary::

    load
    run_mode
    run_mode_dataset
    run
    launch_basic_gui
    build_configuration
    show_versions

Configuration
=============

.. autosummary::

    Configuration
    copy_config_file

Data structures
===============

.. currentmodule:: pyxel.data_structure

.. autosummary::

    Scene
    Photon
    Charge
    Pixel
    Phase
    Signal
    Image
    ArrayBase

Detectors
=========

.. currentmodule:: pyxel.detectors

.. autosummary::

    CCD
    CMOS
    MKID
    APD
    Detector

Attributes
----------

.. autosummary::

    Detector.geometry
    Detector.characteristics
    Detector.scene
    Detector.photon
    Detector.charge
    Detector.pixel
    Detector.signal
    Detector.image
    Detector.data
    Detector.intermediate

Properties
----------

.. currentmodule:: pyxel.detectors

.. autosummary::

    Environment
    Characteristics
    Geometry
    ReadoutProperties
    CCDGeometry
    CMOSGeometry
    MKIDGeometry
    APDCharacteristics
    APDGeometry

Readout time
------------

.. autosummary::

    Detector.set_readout
    Detector.readout_properties
    Detector.time
    Detector.start_time
    Detector.absolute_time
    Detector.time_step
    Detector.times_linear
    Detector.num_steps
    Detector.pipeline_count
    Detector.is_first_readout
    Detector.is_last_readout
    Detector.read_out
    Detector.is_dynamic
    Detector.non_destructive_readout
    Detector.has_persistence
    Detector.persistence
    Detector.numbytes
    Detector.memory_usage

IO / Conversion
---------------

.. autosummary::

    Detector.load
    Detector.save
    Detector.from_hdf5
    Detector.to_hdf5
    Detector.from_asdf
    Detector.to_asdf
    Detector.to_xarray
    Detector.to_dict
    Detector.from_dict

Inputs
======

.. currentmodule:: pyxel

.. autosummary::

    load_image
    load_header
    load_table


Fitness functions
=================

.. currentmodule:: pyxel.calibration

.. autosummary::

    sum_of_abs_residuals
    sum_of_squared_residuals
    reduced_chi_squared

Plotting
========

.. currentmodule:: pyxel.plotting

.. autosummary::

    plot_ptc

Notebook
========

.. currentmodule:: pyxel

General
-------

.. autosummary::

    display_dataset
    display_detector
    display_html
    display_scene

Displaying calibration inputs and outputs
-----------------------------------------

.. autosummary::

    display_calibration_inputs
    display_simulated
    display_evolution
    optimal_parameters
    champion_heatmap

Utility functions
=================

.. currentmodule:: pyxel.util

.. autosummary::

    download_examples
    get_size
    memory_usage_details
    time_pipeline
    fit_into_array
    get_schema

Deprecated / Pending deprecation
================================

.. currentmodule:: pyxel

.. autosummary::

    exposure_mode
    observation_mode
    calibration_mode

Advanced API
============

Pipelines
---------

.. currentmodule:: pyxel.pipelines

.. autosummary::

    DetectionPipeline
    Processor
    ModelGroup
    ModelFunction

Exposure
--------

.. currentmodule:: pyxel.exposure

.. autosummary::

    Exposure

Observation
-----------

.. currentmodule:: pyxel.observation

.. autosummary::

    Observation

Calibration
-----------

.. currentmodule:: pyxel.calibration

.. autosummary::

    Calibration
    CalibrationResult
    MyArchipelago
    Algorithm

Outputs
=======

.. currentmodule:: pyxel.outputs

.. autosummary::

    ExposureOutputs
    ObservationOutputs
    CalibrationOutputs


.. toctree::
   :caption: api reference
   :maxdepth: 1
   :hidden:

   api/run.rst
   api/configuration.rst
   api/datastructures.rst
   api/detectors.rst
   api/detectorproperties.rst
   api/pipelines.rst
   api/exposure.rst
   api/observation.rst
   api/calibration.rst
   api/inputs.rst
   api/outputs.rst
   api/plotting.rst
   api/notebook.rst
   api/util.rst
