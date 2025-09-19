.. _contributing:

=====================
Contributing to Pyxel
=====================

.. contents:: Table of contents:
   :local:

.. note::

  Large parts of this document came from the `Pandas Contributing
  Guide <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_.


Pyxel is built with modularity and flexibility in mind and makes use of
the GitLab infrastructure for version control and to support collaborative
development. This makes it easier for its user & developer
community to directly contribute and expand the framework capabilities by
adding their own models, codes simulating physical processes and effects of
detectors.

`What happens to your code, when you share it? <https://esa.gitlab.io/pyxel/doc/latest/about/FAQ.html#what-happens-to-my-code-when-i-contribute>`_

Where to start?
===============

Pyxel conversation happens in the following places:

#. `GitLab Issue Tracker <https://gitlab.com/esa/pyxel/issues>`_: for discussions around
   new features or established bugs.
#. `Gitter chat <https://gitter.im/pyxel-framework/community>`_: for real-time discussion

For usage questions and bug reports we strongly prefer the use of GitLab issues
over gitter chat.
One of the main reasons for this is that GitLab issues are more easily searchable.
This makes it more efficient for users to locate existing issues.
Gitter chat is generally reserved for community discussion.

All contributions, bug reports, bug fixes, documentation improvements,
and ideas are welcome.

If you are brand new to *Pyxel* or open-source development, we recommend going through
the `GitLab "issues" <https://gitlab.com/esa/pyxel/issues>`_ tab to find issues
that interest you.
There a number of issues listed under `Documentation <https://gitlab.com/esa/pyxel/issues?label_name%5B%5D=documentation>`_
and `good first issue <https://gitlab.com/esa/pyxel/issues?label_name%5B%5D=good+first+issue>`_
where you could start out.
Once you've found an interesting issue, you can return here to get your
development environment setup.

Feel free to ask question on the `Google group <https://groups.google.com/forum/#!forum/pyxel-detector-framework>`_
or on `Gitter <https://gitter.im/pyxel-framework/community>`_

.. _contributing.bug_reports:


Bug reports and enhancement requests
====================================

Bug reports are an important part of making *Pyxel* more stable.
Having a complete bug report will allow others to reproduce the bug and provide
insight into fixing.
See `this stackoverflow article <https://stackoverflow.com/help/mcve>`_
and `this blogspot <http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_
for tips on writing a good bug report.

Trying the bug-producing code out on the *master* branch is often a worthwhile
exercise to confirm the bug still exists. It is also worth searching existing
bug reports and merge requests to see if the issue has already been reported
and/or fixed.

Bug reports must:

1. Include a short, self-contained python snippet reproducing the problem.
   You can format the code nicely by using `GitLab Flavored Markdown
   <https://docs.gitlab.com/ee/user/markdown.html#gitlab-flavored-markdown-gfm>`_::

      ```python
      import pyxel

      cfg = pyxel.load("config.yml")
      ```

2. Include the full version string of *Pyxel* and its dependencies. You can
use the built in function:

.. ipython::
    :okwarning:

    In [1]: import pyxel

    In [2]: pyxel.show_versions()

3. Explain why the current behavior is wrong/not desired and what you expect
   instead.

The issue will be visible to the *Pyxel* community and be open to
comments/ideas from others.

.. _contributing.gitlab:


Setting up a development environment
====================================

Now that you have an issue you want to fix, enhancement to add, or
documentation to improve, you need to learn how to work with GitLab and
the *Pyxel* code base.

This chapter provides instructions for setting up and configuring development environments.

Preliminaries
=============

Basic understanding of how to contribute to Open Source
-------------------------------------------------------

If this is your first open-source contribution, please study one or more of the below resources.

* `How to Get Started with Contributing to Open Source | Video <https://youtu.be/RGd5cOXpCQw>`_
* `Contributing to Open-Source Projects as a New Python Developer | Video <https://youtu.be/jTTf4oLkvaM>`_
* `How to Contribute to an Open Source Python Project | Blog post <https://www.educative.io/blog/contribue-open-source-python-project>`_

.. _contributing.version_control:

Git
---

Version control, Git, and GitLab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To the new user, working with Git is one of the more daunting aspects of
contributing to *Pyxel*.  It can very quickly become overwhelming, but sticking
to the guidelines below will help keep the process straightforward and mostly
trouble free.  As always, if you are having difficulties please feel free
to ask for help.

The code is hosted on `GitLab <https://gitlab.com/esa/pyxel>`_. To
contribute you will need to sign up for a `free GitLab account
<https://gitlab.com/users/sign_in#register-pane>`_. We use `Git <http://git-scm.com/>`_ for
version control to allow many people to work together on the project.

Some great resources for learning Git:

* the `GitLab help pages <https://docs.gitlab.com>`_.
* the `NumPy's documentation <https://numpy.org/doc/stable/dev>`_.
* Matthew Brett's `Pydagogue <https://matthew-brett.github.io/pydagogue>`_.
* Turing Way's guide about `Version Control <https://the-turing-way.netlify.app/reproducible-research/vcs.html>`_.

Getting started with Git
~~~~~~~~~~~~~~~~~~~~~~~~

`GitLab has instructions <https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html>`_
for installing git, setting up your SSH key, and configuring git.
All these steps need to be completed before you can work seamlessly between
your local repository and GitLab.

uv (Unified Python packaging)
-----------------------------

Developing all aspects of Pyxel requires a wide range of packages.
To make this more manageable, `uv <https://docs.astral.sh/uv/>`_ manages the developer experience.

To install `uv <https://docs.astral.sh/uv/>`_,
follow this `guide <https://docs.astral.sh/uv/getting-started/installation/>`_.

For more information, see this `blog post <https://astral.sh/blog/uv-unified-python-packaging>`_.

Installing the project
======================

.. _contributing.forking:

Fork and clone the repository
-----------------------------

The source code for Pyxel is hosted in `GitLab <https://gitlab.com/esa/pyxel>`_.
The first thing you need to do is to fork this repository,
please `follow this guide from gitlab <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html>`_

To create your own fork, go to the `Pyxel project page <https://gitlab.com/esa/pyxel>`_ and
hit the ``Fork`` button (top right, see the following pictures). You have to do this operation only once.

.. figure:: _static/fork_1.png
    :scale: 40%
    :alt: detector
    :align: center

.. figure:: _static/fork_2.png
    :scale: 40%
    :alt: detector
    :align: center

    Example of how to fork Pyxel to your own user space.

After that you will want to clone your fork to your machine.
The following command creates the directory `Pyxel`.

.. code-block:: fish

    git clone https://gitlab.com/YOUR-USER-NAME/pyxel.git
    cd pyxel

Then the following command connects your repository to upstream (main project)
*Pyxel* repository.

.. code-block:: fish

    git remote add upstream https://gitlab.com/esa/pyxel.git

And finally verify the new remote 'upstream' repository:

.. code-block:: fish

    git remote -v


Now you can push/pull your *fork* with ``git push`` and ``git pull``.

A Workflow to keep your fork updated to Pyxel
---------------------------------------------

To keep your fork ``https://gitlab.com/YOUR-USER-NAME/pyxel.git`` updated to
the main repository ``https://gitlab.com/esa/pyxel.git``
follow this `GitLab guide <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html#update-your-fork>`_
or do the following:

1. Make sure that you are on your master branch (from your fork) locally, if not, then
checkout your master branch using this command

    .. code-block:: fish

        git checkout master

2. Then keep your fork updated by merging the new commits from the main repository ``https://gitlab.com/esa/pyxel.git``
to your own local master branch

    .. code-block:: fish

        git fetch upstream master
        git pull upstream master

Now, your local master branch is up-to-date with everything modified upstream (in the
main repository ``https://gitlab.com/esa/pyxel.git``).

This mini-guide is copied from the `workflow to contribute to others project from 'The Turing Way' <https://the-turing-way.netlify.app/reproducible-research/vcs/vcs-github.html?highlight=fork#a-workflow-to-contribute-to-others-github-projects-via-git>`_.

Start developing
================

With 'uv' (Unified Python packaging)
------------------------------------

To start developing with `uv <https://docs.astral.sh/uv/>`_, use the following command:

.. code-block:: fish

    uv sync


.. note::

    * To install Pyxel for Python 3.13, please run the command:

      .. code-block:: fish

          uv sync --python 3.13

    * And to install Pyxel without including its development packages (e.g. ``pytest``, ``mypy``...), use:

      .. code-block:: fish

          uv sync --no-dev

The first time you execute this, a Python Virtual environment ``.venv`` will be created.
Please note that Pyxel will automatically be installed in `editable mode <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_.

**Configuring PyCharm**

If you are using PyCharm, you can set it up to work with ``.venv`` Python virtual environment.
Follow this `guide to configure a virtual environment <https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env>`_
for step-by-step instructions.


With 'pip'
----------

.. _contributing.dev_env:

Creating a development environment
----------------------------------

To test out code changes, you'll need to build *Pyxel* from source, which
requires a Python environment. If you're making documentation changes, you can
skip to :ref:`contributing.documentation` but you won't be able to build the
documentation locally before pushing your changes.

.. _contributing.dev_python:


Creating a Python Environment (conda)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting any development, you'll need to create an isolated Pyxel
development environment:

- Install either `Anaconda3 <https://www.anaconda.com/download/>`_ or `miniconda3
  <https://conda.io/miniconda.html>`_
- Make sure your conda is up to date (launch command ``conda update conda``)
- Make sure that you have :ref:`cloned the repository <contributing.forking>`
- ``cd`` to the *Pyxel* source directory

We'll now kick off a two-step process:

1. Install the build dependencies
2. Build and install Pyxel

.. code-block:: fish

   # Update 'conda' in your base environment
   conda update -n base conda

   # Create the new build environment (once)
   conda env create -f continuous_integration/environment.yml

   # Activate the build environment
   conda activate pyxel-dev

   # or with older versions of Anaconda:
   source activate pyxel-dev

   # Build and install Pyxel in the new environment
   (pyxel-dev) pip install --no-deps -e .

At this point you should be able to import *Pyxel* from your
locally built version:

.. code-block:: fish

   # Start an interpreter
   python
   >>> import pyxel
   >>> pyxel.__version__
   '0.5+0.gcae5a0b'

This will create the new environment, and not touch any of your existing
environments, nor any existing Python installation.

To view your environments:

.. code-block:: fish

      conda info -e

To return to your root environment:

.. code-block:: fish

      conda deactivate

See the full conda docs `here <http://conda.pydata.org/docs>`__.


Creating a Python Environment (pip)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you aren't using conda for your development environment, follow
these instructions:

- You'll need to have at least python3.10 installed on your system.
- Make sure that you have :ref:`cloned the repository <contributing.forking>`
- ``cd`` to the *Pyxel* source directory


.. code-block:: fish

    # Create a virtual environment
    # Use an ENV_DIR of your choice. We'll use ~/virtualenvs/pyxel-dev
    # Any parent directories should already exist
    python3 -m venv ~/virtualenvs/pyxel-dev

    # Activate the virtualenv
    . ~/virtualenvs/pyxel-dev/bin/activate

    # Install the build dependencies
    python -m pip install -r continuous_integration/requirements-dev.txt

    # Build and install Pyxel
    python -m pip install -e .

At this point you should be able to import *Pyxel* from your locally
built version:

.. code-block:: fish

   # Start an interpreter
   python
   >>> import pyxel
   >>> pyxel.__version__
   '1.8+88.g5e2e17dc'


Creating a branch
-----------------

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example:

.. code-block:: fish

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to:

.. code-block:: fish

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to *Pyxel*. You can have many "shiny-new-features"
and switch in between them using the ``git checkout`` command.

To update this branch, you need to retrieve the changes from the master branch:

.. code-block:: fish

    git fetch upstream
    git rebase upstream/master

This will replay your commits on top of the latest *Pyxel* git master.  If this
leads to merge conflicts, you must resolve these before submitting your merge
request.  If you have uncommitted changes, you will need to ``git stash`` them
prior to updating.  This will effectively store your changes and they can be
reapplied after updating.

.. _contributing.documentation:


Contributing to the documentation
=================================

If you're not the developer type, contributing to the documentation is still of
huge value. You don't even have to be an expert on *Pyxel* to do so! In fact,
there are sections of the docs that are worse off after being written by
experts. If something in the docs doesn't make sense to you, updating the
relevant section after you figure it out is a great way to ensure it will help
the next person.


About the *Pyxel* documentation
-------------------------------

The documentation is written in **reStructuredText**, which is almost like
writing in plain English, and built using `Sphinx <http://sphinx.pocoo.org/>`__.
The Sphinx Documentation has an excellent `introduction to reST
<https://sphinx-intro-tutorial.readthedocs.io/en/latest/rst_intro.html>`__. Review the Sphinx docs to perform more
complex changes to the documentation as well.

Some other important things to know about the docs:

- The *Pyxel* documentation consists of two parts: the docstrings in the code
  itself and the docs in this folder ``pyxel/docs/``.

  The docstrings are meant to provide a clear explanation of the usage of the
  individual functions, while the documentation in this folder consists of
  tutorial-like overviews per topic together with some other information
  (what's new, installation, etc).

- The docstrings follow the **Numpy Docstring Standard**, which is used widely
  in the Scientific Python community. This standard specifies the format of
  the different sections of the docstring. See `this document
  <https://numpydoc.readthedocs.io/en/latest/format.html>`_
  for a detailed explanation, or look at some of the existing functions to
  extend it in a similar manner.

- The tutorials make heavy use of the `ipython directive
  <http://matplotlib.org/sampledoc/ipython_directive.html>`_ sphinx extension.
  This directive lets you put code in the documentation which will be run
  during the doc build. For example::

      .. ipython:: python

          x = 2
          x**3

  will be rendered as::

      In [1]: x = 2

      In [2]: x**3
      Out[2]: 8

  Almost all code examples in the docs are run (and the output saved) during the
  doc build. This approach means that code examples will always be up to date,
  but it does make the doc building a bit more complex.

- Our API documentation for models in ``docs/models.rst`` houses the
  auto-generated documentation from the docstrings. For classes, there are
  a few subtleties around controlling which methods and attributes have
  pages auto-generated.


How to build the *Pyxel* documentation
--------------------------------------

Requirements
~~~~~~~~~~~~
Make sure to follow the instructions on :ref:`creating a development
environment above <contributing.dev_env>`, but to build the docs you need
to use the environment file ``continuous_integration/environment.yml``.

.. code-block:: fish

    # Create and activate the docs environment
    conda env create -f continuous_integration/environment.yml
    conda activate pyxel-dev

    # or with older versions of Anaconda:
    source activate pyxel-dev

    # Build and install pyxel
    pip install --no-deps -e .


Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Navigate to your local ``pyxel/docs/`` directory in the console and run:

.. code-block:: fish

    tox -e docs

Then you can find the HTML output in the folder ``pyxels/docs/html/``.

The first time you build the docs, it will take quite a while because it has
to run all the code examples and build all the generated docstring pages.
In subsequent evocations, sphinx will try to only build the pages that
have been modified.

If you want to do a full clean build, do:

.. code-block:: fish

    tox -e docs --recreate

To view the documentation locally, you can also run:

.. code-block:: fish

    tox -e serve-docs



.. _contributing.code:

Contributing to the code base
=============================

Code standards
--------------

Writing good code is not just about what you write. It is also about *how* you
write it. During :ref:`Continuous Integration <contributing.ci>` testing,
several tools will be run to check your code for stylistic errors.
Generating any warnings will cause the test to fail.
Thus, good style is a requirement for submitting code to *Pyxel*.

In addition, because a lot of people use our library, it is important that we
do not make sudden changes to the code that could have the potential to break
a lot of user code as a result, that is, we need it to be as
*backwards compatible* as possible to avoid mass breakages.


.. _contributing.code_formatting:

Code Formatting
~~~~~~~~~~~~~~~
Pyxel uses `tox <https://tox.wiki/en/latest/>`_ and `pre-commit <https://pre-commit.com>`_ to check the code quality.
Both tools can be installed with
``pip``:

.. code-block:: fish

   pip install tox pre-commit

and then run from the root of the Pyxel repository:

.. code-block:: fish

    pre-commit run -a
    tox -p

Backwards Compatibility
~~~~~~~~~~~~~~~~~~~~~~~
Please try to maintain backward compatibility. *Pyxel* has growing number of
users with lots of existing code, so don't break it if at all possible.
If you think breakage is required, clearly state why as part of the merge
request. Also, be careful when changing method signatures and add deprecation
warnings where needed.

Versioning Scheme
~~~~~~~~~~~~~~~~~
Pyxel switch to a new versioning scheme. Pyxel version numbers will be of form x.y.z.

Rules:

- The major release number (x) is incremented if a feature release includes a significant
  backward incompatible change that affects a significant fraction of users.
- The minor release number (y) is incremented on each feature release.
  Minor releases include updated stdlib stubs from typeshed.
- The point release number (z) is incremented when there are fixes only.

Pyxel doesn't use SemVer anymore, since most minor releases have at least minor backward incompatible changes.

Any significant backward incompatible change must be announced in the
`changelog <https://esa.gitlab.io/pyxel/doc/stable/references/changelog.html>`_ for the previous feature release,
before making the change.

.. _contributing.documenting_your_code:

Documenting your code
---------------------

Changes should be reflected in the release notes located in ``CHANGELOG.md``.
This file contains an ongoing change log for each release.  Add an entry to
this file to document your fix, enhancement or (unavoidable) breaking change.
Make sure to include the GitLab issue number when adding your entry (using
``#1234``, where ``1234`` is the issue/merge request number).

If your code is an enhancement, it is most likely necessary to add usage
examples to the existing documentation.  This can be done following the section
regarding documentation :ref:`above <contributing.documentation>`.


Testing
-------

.. _contributing.ci:

Testing With Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuous Integration (CI) automatically integrate code changes from multiple stakeholders
in a single software project. It allows developers to frequently contribute code changes
to a central repository where builds and tests are then executed.
Automated tools are used to verify the correctness of new code before the integration.
The version control system in Git is also supported by other checks such as automated code
quality tests, syntax style checking tools and more (see `GitLab CI <https://about.gitlab.com/features/continuous-integration/>`_).
For example, the CI tool `tox <https://tox.wiki/en/latest/>`_ aims to automate and
standardize testing in Python. It (*tox*) is a generic virtual environment management and
test command line tool you can use for:

* checking your package builds and installs correctly under different environments (such
  as different Python implementations, versions or installation dependencies),
* running your tests in each of the environments with the test tool of choice,
* acting as a frontend to continuous integration servers and merging CI and shell-based
  testing.

.. _contributing.test-driven-development-code-writing:

Test-driven development/code writing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Pytest* is serious about testing and strongly encourages contributors to embrace
`test-driven development (TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`_.
This development process "relies on the repetition of a very short development
cycle: first the developer writes an (initially failing) automated test case
that defines a desired improvement or new function, then produces the minimum
amount of code to pass that test. So, before actually writing any code, you
should write your tests.  Often the test can be taken from the original GitLab
issue. However, it is always worth considering additional use cases and writing
corresponding tests.

Adding tests is one of the most common requests after code is pushed
to *Pytest*. Therefore, it is worth getting in the habit of writing tests
ahead of time so this is never an issue.

Like many packages, *Pytest* uses `pytest <http://doc.pytest.org/en/latest/>`_
and the convenient extensions in `numpy.testing <http://docs.scipy.org/doc/numpy/reference/routines.testing.html>`_.


Writing tests
~~~~~~~~~~~~~

All tests should go into the ``tests`` directory of the specific package.
This folder contains many current examples of tests, and we suggest looking to
these for inspiration.

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

The tests can then be run directly inside your Git clone (without having
to install *Pyxel*) by typing:

.. code-block:: fish

    pytest

The tests suite is exhaustive and takes a few minutes.  Often it is
worth running only a subset of tests first around your changes before
running the entire suite.

The easiest way to do this is with:

.. code-block:: fish

    pytest tests/path/to/test.py -k regex_matching_test_name

Or with one of the following constructs:

.. code-block:: fish

    pytest tests/[test-module].py
    pytest tests/[test-module].py::[TestClass]
    pytest tests/[test-module].py::[TestClass]::[test_method]

Using `pytest-xdist <https://pypi.python.org/pypi/pytest-xdist>`_, one can
speed up local testing on multicore machines. To use this feature, you will
need to install `pytest-xdist` via:

.. code-block:: fish

    pip install pytest-xdist


Then, run pytest with the optional -n argument:

.. code-block:: fish

    pytest -n 4

This can significantly reduce the time it takes to locally run tests before
submitting a pull request.

For more, see the `pytest <http://doc.pytest.org/en/latest/>`_ documentation.


Running the performance test suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To run performance test(s)/benchmark(s), check the
`repository "Pyxel benchmarks" <https://gitlab.com/esa/pyxel-benchmarks>`_.
To visualize the performance test(s)/benchmark(s), see
`benchmarks <https://esa.gitlab.io/pyxel/benchmarks/>`_.

.. _contributing.yourchanges:

Contributing your changes to *Pyxel*
====================================

Committing your code
--------------------

Keep style fixes to a separate commit to make your pull request more readable.

Once you've made changes, you can see them by typing:

.. code-block:: fish

    git status

If you have created a new file, it is not being tracked by git.
Add it by typing:

.. code-block:: fish

    git add path/to/file-to-be-added.py

Doing 'git status' again should give something like:

.. code-block:: fish

    # On branch shiny-new-feature
    #
    #       modified:   /relative/path/to/file-you-added.py
    #

The following defines how a commit message should be structured:

    * A subject line with `< 72` chars.
    * One blank line.
    * Optionally, a commit message body.

Please reference the relevant GitLab issues in your commit message
using ``#1234``.


Now you can commit your changes in your local repository:

.. code-block:: fish

    git commit -m


Pushing your changes
--------------------

When you want your changes to appear publicly on your GitLab page, push your
forked feature branch's commits:

.. code-block:: fish

    git push origin shiny-new-feature

Here ``origin`` is the default name given to your remote repository on GitLab.
You can see the remote repositories:

.. code-block:: fish

    git remote -v

If you added the upstream repository as described above you will see something
like:

.. code-block:: fish

    origin    https://gitlab.com/your-user-name/pyxel.git (fetch)
    origin    https://gitlab.com/your-user-name/pyxel.git (push)
    upstream  https://gitlab.com/esa/pyxel.git (fetch)
    upstream  https://gitlab.com/esa/pyxel.git (push)

Now your code is on GitLab, but it is not yet a part of the *Pyxel* project.
For that to happen, a merge request needs to be submitted on GitLab.


Review your code
----------------

When you're ready to ask for a code review, file a merge request.
Before you do, once again make sure that you have followed all the guidelines
outlined in this document regarding code style, tests, performance tests,
and documentation.
You should also double check your branch changes against the branch
it was based on:

#. Navigate to your repository on GitLab -- https://gitlab.com/your-user-name/pyxel
#. Click on ``Repository`` and then ``Branches``
#. Click on the ``Compare`` button for your feature branch
#. Select the ``base`` and ``compare`` branches, if necessary.
   This will be ``master`` and ``shiny-new-feature``, respectively.

Finally, make the merge request
-------------------------------

If everything looks good, you are ready to make a merge request.
A merge request is how code from a local repository becomes available to
the GitLab community and can be looked at and eventually merged into
the master version.  This merge request and its associated changes
will eventually be committed to the master branch and available in the next
release. To submit a merge request:

1. Navigate to your repository on GitLab
2. Click on the ``Merge Requests`` and the button ``New merge request``.
3. You can then select the branch to merge from your fork to ``esa/pyxel`` (see following picture).

.. figure:: _static/new_merge_request.png
    :scale: 40%
    :alt: detector
    :align: center

    Create a new merge request.

4. Write a description of your changes in the ``Discussion`` tab.
5. Click ``Create Merge Request`` and check if you have fulfilled all requirements from the :ref:`"Merge request checklist" <contributing.mergechecklist>`.

This request then goes to the repository maintainers, and they will review
the code. If you need to make more changes, you can make them in your branch,
add them to a new commit, push them to GitLab, and the merge request will
be automatically updated.  Pushing them to GitLab again is done by:

.. code-block:: fish

    git push origin shiny-new-feature

This will automatically update your merge request with the latest code
and restart the :ref:`Continuous Integration <contributing.ci>` tests.


Delete your merged branch (optional)
------------------------------------

Once your feature branch is accepted into upstream, you'll probably want
to get rid of the branch. First, merge upstream master into your branch
so git knows it is safe to delete your branch:

.. code-block:: fish

    git fetch upstream
    git checkout master
    git merge upstream/master

Then you can do:

.. code-block:: fish

    git branch -d shiny-new-feature

Make sure you use a lower-case ``-d``, or else git won't warn you if your
feature branch has not actually been merged.

The branch will still exist on GitLab, so to delete it there do:

.. code-block:: fish

    git push origin --delete shiny-new-feature

.. _contributing.mergechecklist:

Merge Request checklist
-----------------------

- **Properly comment and document your code.** See :ref:`"Documenting your
  code" <contributing.documenting_your_code>`.
- **Test that the documentation builds correctly** by typing ``tox -e docs``.
  This is not strictly necessary, but this may be easier than waiting for CI
  to catch a mistake.
  See :ref:`"Contributing to the documentation" <contributing.documentation>`.
- **Test your code**.

    - Write new tests if needed. See :ref:`"Test-driven development/code
      writing" <contributing.test-driven-development-code-writing>`.
    - Test the code using `Pytest <http://doc.pytest.org/en/latest/>`_.
      Running all tests (type ``pytest`` in the root directory) takes a while,
      so feel free to only run the tests you think are needed based on
      your Merge Request (example: ``pytest tests/test_model_xxx.py``).
      CI will catch any failing tests.

- **Properly format your code** and verify that it passes the formatting guidelines
  set by `tox <https://tox.wiki/en/latest/>`_ and `pre-commit <https://pre-commit.com>`_ to check the code quality.
  See :ref:`"Code formatting" <contributing.code_formatting>`.


  Run from the root of the Pyxel repository:

  .. code-block:: fish

    pre-commit run -a
    tox -p

- **Push your code and** `create a Merge Request on GitLab <https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html>`_.
- **Use a helpful title for your merge request** by summarizing the main contributions rather than using the latest commit message.
  If this addresses an `issue <https://gitlab.com/esa/pyxel/issues>`_, please `reference it <https://docs.gitlab.com/ee/user/project/issues/crosslinking_issues.html>`_.
