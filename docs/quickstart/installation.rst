Installation
====================
It's generally good practice to install this sort of tool in its own virtual environment (here's a `virtual environment tutorial <https://www.w3schools.com/python/python_virtualenv.asp>`_).  Starlord is not yet listed on PyPy, so you'll need to install it from the Github repository.  This can be done with pip:

.. code-block:: console

    pip install git+https://github.com/dpthorngren/Starlord.git#egg=starlord

Alternatively, if you'd like to download the git repository somewhere specific, you can go to that directory and use:

.. code-block:: console

    git clone git@github.com:dpthorngren/Starlord.git
    cd Starlord
    pip install .

If you'd like to run the tests, install the test dependencies with ``pip install .[develop]``, then run them with ``pytest`` from the Starlord directory.  Either way, you should now be able to run Starlord with ``starlord`` in the command line.

.. tip::

   You can see a list of starlord command line options and their use with ``starlord --help``, and the Starlord :doc:`../ref/index` can also be viewed with ``python -m pydoc starlord``.
