.. _examples:

======================
Tutorials and examples
======================

We recommend you to start with the tutorial available in the form of Jupyter notebooks.
It covers all the basics, the four running modes and adding a new model. Apart from the tutorial,
more examples on running modes and different models are also available. See below for a full list.

All tutorials and examples can be found in a separate public repository
`Pyxel Data <https://gitlab.com/esa/pyxel-data>`_, to access the corresponding Jupyter book click on the link below.
Please note, that the Jupyter book is just for viewing the examples and not interactive.

**Contact**: pyxel@esa.int

.. button-link:: https://esa.gitlab.io/pyxel-data/intro.html
    :color: primary
    :outline:
    :expand:
    :ref-type: any

    To tutorials and examples

If you want to change the inputs and see the results immediately,
you need to a :ref:`install` of Pyxel and download the `Pyxel Data <https://gitlab.com/esa/pyxel-data>`_ repository
or launch a live session on Binder.

Once you’ve installed Pyxel, the example repository can be either downloaded directly by clicking on button download
or using Pyxel by running the command:

.. code-block:: console

    pyxel-sim download-examples

    or

    python -m pyxel-sim download-examples

Now you can launch JupyterLab to explore them:

.. code-block:: console

    cd pyxel-examples

    jupyter lab

You can run also tutorials and examples without prior installation of Pyxel in a live session here: |Binder|

.. |Binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gl/esa%2Fpyxel-data/HEAD?urlpath=lab


Generic detector pipelines
--------------------------

The Pyxel model library contains models for various types of detectors.
Not all models can be used with all of the detector types
and some specific models are only to be used with a single type of detector.
For this reason and to help new users and non-experts,
generic configuration file templates for different detectors have been included in the Pyxel Data example repository,
together with corresponding Jupyter notebooks.
They include detector properties and pipelines with detector-appropriate sets of models,
pre-filled with realistic model argument values.
They provide a good starting point for simulations of specific detectors and later customization
or iteration with detector engineers and experts.
The generic pipelines are now available for the following types
of detectors: generic :term:`CCD`, generic :term:`CMOS`, Teledyne HxRG, Microwave kinetic-inductance detector (:term:`MKID`)
and Avalanche Photo Diode (:term:`APD`) array detector based on Leonardo’s Saphira detector.
