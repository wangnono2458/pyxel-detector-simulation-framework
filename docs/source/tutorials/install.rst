.. _install:

============
Installation
============

There are many ways to install Pyxel. On this page we list the most common ones.
In general **we recommend to use virtual environments** when using Pyxel.
This ways you have full control over addition packages that you may use in your analysis
and your work with well defined computing environments.

If you want to learn about using virtual environments see :ref:`Virtual Environments <virtualenvs>`.
You can also install :ref:`Pyxel for development <contributing.gitlab>`.

Don't hesitate to create a `GitLab issues <https://gitlab.com/esa/pyxel/-/issues>`_
to report bugs, ask for new functionalities, ... .

**Contact**: pyxel@esa.int


..
    Pyxel can be installed with `conda <https://docs.conda.io/>`_,
    `pip <https://pip.pypa.io/>`_ or from source.

    If you want to have a full installation of Pyxel, then the recommended installation
    method is to use `conda <https://docs.conda.io/>`__ into a conda environment.

    The following instructions are valid for MacOS, Windows and Linux.


    .. important::
        Because of its optional and required dependencies, Pyxel is not compatible with
        all versions of Python.

        You can install Pyxel with `pygmo <https://esa.github.io/pygmo2/>`_ only for
        **Python 3.7** and **Python 3.8** (not Python 3.9+).

        If you don't use `pygmo <https://esa.github.io/pygmo2/>`_ then you can Pyxel with
        **Python 3.7**, **Python 3.8** and **Python 3.9** (not Python 3.10+).


    .. warning::
        It is **strongly** encouraged to install optional package
        `pygmo <https://esa.github.io/pygmo2/>`_ with ``conda`` rather than ``pip``.
        See `here <https://esa.github.io/pygmo2/install.html#pip>`_ for more information.

        Moreover, only the binaries of ``pygmo`` for Linux (not MacOS or Windows)
        are available on ``pip``.
        The binaries of ``pygmo`` for MacOS, Windows and Linux are only available
        on Conda 64bit (**not 32bit**).


Using Anaconda / Miniconda
==========================

The easiest way to install Pyxel for Linux, MacOS and Windows is
to install `Anaconda <https://www.anaconda.com/download>`_
or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ Python distribution.
The packages are available from the
`conda-forge <https://anaconda.org/conda-forge/pyxel-sim>`_.

.. note::

    It is **strongly** encouraged to install Pyxel with ``conda`` (or ``mamba``) rather than ``pip`` because
    of the optional dependency `pygmo <https://esa.github.io/pygmo2/>`_.
    `pygmo <https://esa.github.io/pygmo2/>`_ is exclusively used for the calibration mode.
    See `here <https://esa.github.io/pygmo2/install.html#pip>`_ for more information.

.. important::

    You **must** install a 64-bit version of `Anaconda <https://www.anaconda.com/download>`_
    or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_


To install the latest stable version of Pyxel, execute this command in a terminal:

.. code-block:: bash

    conda install -c conda-forge pyxel-sim

.. note::

    It is recommended to install Pyxel in its own dedicated Conda environment.
    For more information, click here :ref:`conda_envs`.

To update an existing installation you can use:

.. code-block:: bash

    conda update pyxel-sim

.. note::

    For now, it's not possible to install a recent version of
    `lacosmic <https://lacosmic.readthedocs.io/en/stable/api/lacosmic.lacosmic.html#lacosmic.lacosmic>`__
    for all platforms directly from ``conda`` or ``mamba``.
    The user **must** install ``lacosmic`` manually (in the current conda environment) with the
    command ``pip``:

    .. code-block:: bash

        pip install lacosmic


Using Mamba
===========

Alternatively, you can use `Mamba <https://mamba.readthedocs.io/>`_ for the installation.
Mamba is an alternative package manager that support most of conda's command but offers
higher installation speed and more reliable environment solutions.
To install ``mamba`` in the Conda base environment:

.. code-block:: bash

    conda install mamba -n base -c conda-forge

then:

.. code-block:: bash

    mamba install -c conda-forge pyxel-sim

Mamba supports of the commands that are available for conda.
So updating and installing specific versions works the same way
as above except for replacing the ``conda`` with the ``mamba`` command.

.. note::

    For now, it's not possible to install a recent version of
    `lacosmic <https://lacosmic.readthedocs.io/en/stable/api/lacosmic.lacosmic.html#lacosmic.lacosmic>`__
    for all platforms directly from ``conda`` or ``mamba``.
    The user **must** install ``lacosmic`` manually (in the current conda environment) with the
    command ``pip``:

    .. code-block:: bash

        pip install lacosmic

Pip
===

To install the latest Pyxel **stable** version
(see `Pyxel page on PyPi <https://pypi.org/project/pyxel-sim>`_)
using `pip <https://pip.pypa.io>`_:



Full installation
-----------------

.. note::

    It is recommended to install Pyxel in its own dedicated Python's virtual environment.
    For more information, click here :ref:`venv_envs`.


To install all optional dependencies of Pyxel, you must run the command:

.. code-block:: bash

   pip install pyxel-sim[all]    # Install everything (only on Linux !)

To install only the optional dependencies for the models, you can run:

.. code-block:: bash

   pip install pyxel-sim[model]  # Install all extra dependencies
                                 # for models (poppy, lacosmic)


.. warning::
    Library ``pygmo2`` is only available for Linux on PyPi.

    If you want to use the calibration mode on Windows or MacOS, you must
    install Pyxel with ``conda``.

Updating
--------

To update Pyxel with ``pip``, you can use the following command:

.. code-block:: bash

    pip install -U pyxel-sim


Install from source
===================

To install Pyxel from source, clone the repository from the
`Pyxel GitLab repository <https://gitlab.com/esa/pyxel>`_.

.. code-block:: bash

    # Get source code
    git clone https://gitlab.com/esa/pyxel.git
    cd pyxel
    python install -m pip install .

You can install all dependencies as well:

.. code-block:: bash

    python -m pip install ".[all]"


Or do a developer install by using the `-e` flag (For more information
see :ref:`contributing.dev_env` from the page :ref:`contributing`)

.. code-block:: bash

    python -m pip install -e .


Verify the installation
=======================

You can verify that Pyxel is installed with the following command:

.. code-block:: bash

    python -c "import pyxel; pyxel.show_versions()"


Dependencies
============

Required dependencies
---------------------

Pyxel has the following **mandatory** dependencies:

=================================================================================== ========================= ===================================
Package                                                                             Minimum supported version Notes
=================================================================================== ========================= ===================================
`python <https://www.python.org>`_                                                  3.10
`numpy <https://numpy.org>`_                                                        1.24
`xarray <http://xarray.pydata.org/>`_                                               2024.10.0                 API for N-dimensional data
`astropy <https://www.astropy.org>`_                                                4.3
`pandas <https://pandas.pydata.org>`_                                               1.5
`numba <https://numba.pydata.org>`_                                                 0.56.4                    Performance using a JIT compiler
`scipy <https://scipy.org>`_                                                        1.10                      Miscellaneous statistical functions
`holoviews <https://holoviews.org>`_                                                1.15
`matplotlib <https://matplotlib.org>`_                                              3.6                       Plotting library
`bokeh <http://bokeh.org>`_                                                         3.3.0
`dask <https://dask.org>`_
`tqdm <https://tqdm.github.io>`_
=================================================================================== ========================= ===================================

Optional dependencies
---------------------

Pyxel has many optional dependencies for specific functionalities.
If an optional dependency is not installed, Pyxel will raise an ``ImportError`` when
the functionality requiring that dependency is called.

If using pip, optional pyxel dependencies can be installed as optional extras
(e.g. ``pyxel-sim[model,calibration]``).
All optional dependencies can be installed with ``pip install "pyxel-sim[all]"``,
and specific sets of dependencies are listed in the sections below.

Models dependencies
~~~~~~~~~~~~~~~~~~~

Installable with ``pip install "pyxel-sim[model]"``.

======================================================================================================= =============== ==============================================================
Package                                                                                                 Minimum version Notes
======================================================================================================= =============== ==============================================================
`sep <https://sep.readthedocs.io>`_                                                                                     For model ``extract_roi_to_xarray``
`poppy <https://poppy-optics.readthedocs.io/>`_                                                         1.1.0           For models ``optical_psf`` and ``optical_psf_multi_wavelength``
`lacosmic <https://lacosmic.readthedocs.io/en/stable/api/lacosmic.lacosmic.html#lacosmic.lacosmic>`__                   For model ``remove_cosmic_rays``
======================================================================================================= =============== ==============================================================

.. note::
    Optional package
    `lacosmic <https://lacosmic.readthedocs.io/en/stable/api/lacosmic.lacosmic.html#lacosmic.lacosmic>`__ is not available
    on ``conda``, only on the ``PyPI`` repository.


Calibration mode
~~~~~~~~~~~~~~~~

To use the calibration mode, you must use ``pip install "pyxel-sim[calibration]"``.

=========================================== ===============
Package                                     Minimum version
=========================================== ===============
`pygmo <https://esa.github.io/pygmo2/>`_    2.16.1
=========================================== ===============


Extra data sources
~~~~~~~~~~~~~~~~~~

Installable with ``pip install "pyxel-sim[io]"``.

======================================================= =============== ===============================================
Package                                                 Minimum version Notes
======================================================= =============== ===============================================
`h5py <https://www.h5py.org>`_
`netcdf4 <https://unidata.github.io/netcdf4-python/>`_
`fsspec <https://filesystem-spec.readthedocs.io>`_      2021            Handling files aside from simple local and HTTP
======================================================= =============== ===============================================



..
    Python
    ~~~~~~

    Before you got any further, make sure you've got Python 3.7 or newer available
    from your command line.

    You can check this by simply running:

    .. code-block:: bash

      $ python3 --version
      Python 3.7.2

      or

      $ python3.7 --version
      Python 3.7.2


    On Windows, you can also try:

    .. code-block:: bash

     $ py -3 --version
     Python 3.7.2

     or

     $ py -3.7 --version
     Python 3.7.2

    .. note::

      Do not use command ``python``, you should use a command like ``pythonX.Y``.
      For example, to start Python 3.7, you use the command ``python3.7``.


..
    Pip
    ~~~

    Furthermore, you'll need to make sure pip is installed with a recent version.
    You can check this by running:

    .. code-block:: bash

      $ python3.7 -m pip --version
      pip 19.1.1

    .. note::

      Do not use command ``pip`` but ``python -m pip``.
      For example, to start ``pip`` for Python 3.7, you use the
      command ``python3.7 -m pip``.

    You can find more information about installing packages
    at this `link <https://packaging.python.org/installing/>`_.


..
    Install from source
    ===================

    Get source code
    ~~~~~~~~~~~~~~~

    First, get access to the `Pyxel GitLab repository <https://gitlab.com/esa/pyxel>`_
    from maintainers (pyxel at esa dot int).

    If you can access it, then clone the GitLab repository to your computer
    using ``git``:

    .. code-block:: bash

        $ git clone https://gitlab.com/esa/pyxel.git


..
    Install requirements
    ~~~~~~~~~~~~~~~~~~~~

    After cloning the repository, install the dependency provided together
    with Pyxel using ``pip``:


    .. code-block:: bash

      $ cd pyxel
      $ python3.7 -m pip install -r requirements.txt

    .. note::
      This command installs all packages that cannot be found in ``pypi.org``.
      This step will disappear for future versions of ``pyxel``.

    .. important::
      To prevent breaking any system-wide packages (ie packages installed for all users)
      or to avoid using command ``$ sudo pip ...`` you can
      do a `user installation <https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.

      With the command: ``$ python3.7 -m pip install --user -r requirements.txt``

..
    Install Pyxel
    ~~~~~~~~~~~~~

    To install ``pyxel`` use ``pip`` locally, choose one from
    the 4 different options below:


    .. code-block:: bash

      $ python3.7 -m pip install -e ".[all]"            # Install everything (recommended)
      $ python3.7 -m pip install -e ".[calibration]"    # Install dependencies for 'calibration mode' (pygmo)
      $ python3.7 -m pip install -e ".[model]"          # Install dependencies for optional models (poppy, lacosmic)
      $ python3.7 -m pip install -e .                   # Install without any optional dependencies


    ..
      To install ``pyxel`` use ``pip`` locally, choose one from the 4 different options below:

        * To install ``pyxel`` and all the optional dependencies (recommended):

        .. code-block:: bash

          $ python3.7 -m pip install -e ".[all]"

        * To install ``pyxel`` and the optional dependencies for *calibration mode* (``pygmo``):

        .. code-block:: bash

          $ python3.7 -m pip install -e ".[calibration]"

        * To install ``pyxel`` and the optional models (``poppy``, ``lacosmic``):

        .. code-block:: bash

          $ python3.7 -m pip install -e ".[model]"

        * To install ``pyxel`` without any optional dependency:

        .. code-block:: bash

          $ python3.7 -m pip install -e .


    .. important::
      To prevent breaking any system-wide packages (ie packages installed for all users)
      or to avoid using command ``$ sudo pip ...`` you can do a `user installation <https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.
      Whenvever you see the command ``$ python3.7 -m pip install ...`` then replace it
      by the command ``$ python3.7 -m pip install --user ...``.

      If ``pyxel`` is not available in your shell after installation, you will need to add
      the `user base <https://docs.python.org/3/library/site.html#site.USER_BASE>`_'s binary
      directory to your PATH.

      On Linux and MacOS the user base binary directory is typically ``~/.local``.
      You'll need to add ``~/.local/bin`` to your PATH.
      On Windows the user base binary directory is typically
      ``C:\Users\Username\AppData\Roaming\Python36\site-packages``.
      You will need to set your PATH to include
      ``C:\Users\Username\AppData\Roaming\Python36\Scripts``.
      you can find the user base directory by running
      ``python3.7 -m site --user-base`` and adding ``bin`` to the end.


    After the installation steps above,
    see :ref:`here how to run Pyxel <running_modes>`.

..
    Install from PyPi
    -----------------

    TBW.


    To upgrade ``pyxel`` to the latest version:

    TBW.

..
    Install with Anaconda
    ---------------------

    TBW.

    .. note::
      If a package is not available in any PyPI server for your OS, because
      you are using Conda or Anaconda Python distribution, then you might
      have to download the Conda compatible whl file of some dependencies
      and install it manually with ``conda install``.

      If you use OSX, then you can only install ``pygmo`` with Conda.

..
    Using Docker
    -------------

    TBW.

..
    Installation with Anaconda
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    First install the `Anaconda distribution <https://www.anaconda.com/distribution/>`_
    then check if the tool ``conda`` is correctly installed:

    .. code-block:: bash

      $ conda info

    The second step is to create a new conda environment `pyxel-dev` and
    to install the dependencies with ``conda`` and ``pip``:

    .. code-block:: bash

      $ cd pyxel

      Create a new conda environment 'pyxel-dev'
      and install some dependencies from conda with `continuous_integration/environment.yml`
      $ conda env create -f continuous_integration/environment.yml

      Display all conda environments (only for checking)
      $ conda info --envs

      Activate the conda environment 'pyxel-dev'
      $ (pyxel-dev) conda activate pyxel-dev

      Install the other dependencies not installed by conda
      $ (pyxel-dev) pip install -r requirements.txt


    Then install ``pyxel`` in the conda environment:

    .. code-block:: bash

      $ (pyxel-dev) cd pyxel
      $ (pyxel-dev) pip install --no-deps -e .

    More about the conda environments (only for information):

    .. code-block:: bash

      Deactivate the environment
      $ conda deactivate

      Remove the conda environment 'pyxel-dev'
      $ conda remove --name pyxel-dev --all

    After the installation steps above,
    see :ref:`here how to run Pyxel <running_modes>`.

..
    Using Docker
    -------------

    Using Docker, you can just download the Pyxel Docker image and run it without
    installing Pyxel.

    How to run a Pyxel container with Docker:

    Login:

    .. code-block:: bash

      docker login gitlab.esa.int:4567

    Pull latest version of the Pyxel Docker image:

    .. code-block:: bash

      docker pull gitlab.esa.int:4567/sci-fv/pyxel

    Run Pyxel Docker container with GUI:

    .. code-block:: bash

      docker run -p 9999:9999 \
                 -it gitlab.esa.int:4567/sci-fv/pyxel:latest \
                 --gui True

    Run Pyxel Docker container in batch mode (without GUI):

    .. code-block:: bash

      docker run -p 9999:9999 \
                 -v C:\dev\work\docker:/data \
                 -it gitlab.esa.int:4567/sci-fv/pyxel:latest \
                 -c /data/settings_ccd.yaml \
                 -o /data/result.fits

    List your running Docker containers:

    .. code-block:: bash

      docker ps

    After running Pyxel container you can access it:

    .. code-block:: bash

      docker exec -it <CONTAINER_NAME> /bin/bash
