=============
Running Pyxel
=============

Pyxel can be run either from the command line or used as a library, such as in Jupyter notebooks.

.. note::

   After installing Pyxel, you can directly download examples files using one of the following commands:

   .. code-block:: bash

       pyxel-sim download-examples

   or

   .. code-block:: bash

       python -m pyxel-sim download-examples


   Alternatively, you can use `uv <https://docs.astral.sh/uv/>`_ (see the installation
   guide `here <https://docs.astral.sh/uv/#getting-started>`_) to download the examples/tutorials:

   .. code-block:: bash

       uvx pyxel-sim download-examples


   These examples will be saved in a new folder called ``pyxel-examples``.
   More information can be found in the :doc:`examples` documentation.


Running Pyxel from command line
===============================

To run Pyxel locally, simply use the command-line:

.. code-block:: bash

    pyxel-sim run input.yaml

or

.. code-block:: bash

    python -m pyxel-sim run input.yaml


Alternatively, with `uv <https://docs.astral.sh/uv/>`_ (see `here <https://docs.astral.sh/uv/#getting-started>`_)

.. code-block:: bash

    uvx --with pyxel-sim[model] pyxel-sim run input.yaml


Usage:

.. code-block:: bash

    Usage: pyxel-sim run [OPTIONS] CONFIG

      Run Pyxel with a YAML configuration file.

    Options:
      --override TEXT  Override entries from the YAML configuration file. This
                        parameter can be repeated.

                        Example:

                        --override exposure.outputs.output_folder=new_folder
      -v, --verbosity     Increase output verbosity (-v/-vv/-vvv)  [default: 0]
      -s, --seed INTEGER  Random seed for the framework.
      --help              Show this message and exit.

where

========================  =======================================  ========
``CONFIG``                defines the path of the input YAML file  required
``-s`` / ``--seed``       defines a seed for random number         optional
                          generator
``-v`` / ``--verbosity``  increases the output verbosity (-v/-vv)  optional
``-V`` / ``--version``    prints the version of Pyxel              optional
========================  =======================================  ========

Running Pyxel in jupyter notebooks
==================================

An example of running Pyxel as a library:

.. code-block:: python

    import pyxel

    configuration = pyxel.load("configuration.yaml")

    pyxel.run_mode(configuration)

.. Note::
   You need install a Jupyter Server yourself (e.g. Jupyter Notebook, Jupyter Lab, Jupyter Hub...).

   If you want to display all intermediate steps computed by function ``pyxel.run_mode``, you can check this link:
   `Is there a way to display all intermediate steps when a pipeline is executed ? <https://esa.gitlab.io/pyxel/doc/stable/about/FAQ.html#is-there-a-way-to-display-all-intermediate-steps-when-a-pipeline-is-executed>`_


Running Pyxel from a Docker container
=====================================

If you want to run Pyxel in a Docker container, you must first get the source code
from the `Pyxel GitLab repository <https://gitlab.com/esa/pyxel>`_.

.. code-block:: console

    git clone https://gitlab.com/esa/pyxel.git
    cd pyxel


Build an image
--------------

.. tab:: docker-compose

    .. code-block:: console

        # Create docker image 'pyxel_pyxel'
        docker-compose build

.. tab:: only docker

    .. code-block:: console

        # Create docker image 'pyxel'
        docker build --tag pyxel .


Create and start the container
------------------------------

Run Pyxel with a Jupyter Lab server from a new docker container:

.. tab:: docker-compose

    .. code-block:: console

        # Create and start a new container 'pyxel_pyxel_1'
        docker-compose up --detach

.. tab:: only docker

    .. code-block:: console

        # Create and start new container 'pyxel_dev' from image 'pyxel'
        docker create -p 8888:8888 pyxel --name pyxel_dev
        docker start pyxel_dev

Stop and remove the container
-----------------------------

Stop and remove a running Pyxel container.

.. tab:: docker-compose

    .. code-block:: console

        # Stop and remove container 'pyxel_pyxel_1'
        docker-compose down

.. tab:: only docker

    .. code-block:: console

        # Stop and remove container 'my_pyxel'
        docker stop my_pyxel
        docker rm my_pyxel

Check if the container is running
----------------------------------

List running containers.

.. tab:: docker-compose

    .. code-block:: console

        docker-compose ps


.. tab:: only docker

    .. code-block:: console

        docker ps


Get logs
--------

View output from the Pyxel container.

.. tab:: docker-compose

    .. code-block:: console

        # Get logs from container 'pyxel_pyxel_1'
        docker-compose logs -f


.. tab:: only docker

    .. code-block:: console

        # Get logs from container 'my_pyxel'
        docker logs -f my_pyxel
