.. _channels:

========
Channels
========

In Detectors, especially CCDs and CMOSs image sensors, the concept of **channels** is used to enable
parallel readout of the image data.
This improve readout speed and efficiency by dividing the sensor into multiple regions, each processed independently.

What is a Channel ?
===================

A **channel** refers to subdivided region of the detector array.
Detectors can be split vertically, horizontally, or both, resulting in a grid
of channels.

Each of these channels can be read out via a dedicated amplifier or readout node.

For example:

- A detector split **vertically** into two channels will have a left and a right region.
- A detector split **horizontally** into two channels will have a top and a bottom region.
- A detector split both **vertically and horizontally** results in four quadrants or channels

Readout Positions
=================

Each channel has a **readout position** defined to determine the direction from which the pixel
data is read.

The supported readout positions are:

- ``top-left``
- ``top-right``
- ``bottom-left``
- ``bottom-right``

These positions are essential when modeling charge transfer and readout timings,
as they influence how pixels are clocked out during readout.

Example
=======

.. figure:: _static/channels.png
    :width: 800px
    :alt: channels
    :align: center

    Example with four channels.

This figure above shows a typical configuration of a detector with four channels,
each associated with its own readout node at the respective corner.

Use in simulation
=================

Channel configuration is critical for accurately modeling detector behaviour,
for instance for the `DC crosstalk <https://esa.gitlab.io/pyxel/doc/stable/references/model_groups/charge_measurement_models.html#dc-crosstalk>`_
and `AC crosstalk <https://esa.gitlab.io/pyxel/doc/stable/references/model_groups/charge_measurement_models.html#ac-crosstalk>`_
models.

Example configuration snippet (YAML)
====================================

Here's an example of how channel configuration might appear in a YAML-based file:


.. code-block:: yaml

 geometry:
    row: 1028
    col: 1024
    channels:
      matrix: [[OP9, OP13],
               [OP1, OP5 ]]
      readout_position:
        - OP9:  top-left
        - OP13: top-left
        - OP1:  bottom-left
        - OP5:  bottom-left
