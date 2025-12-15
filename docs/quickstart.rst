Quickstart
====================

This will be a guide... eventually.

Installation
--------------------
It's generally good practice to install this sort of tool in its own virtual environment.

.. code:: bash

    pip install git+https://github.com/dpthorngren/Starlord.git#egg=starlord

Alternatively:

.. code:: bash

    git clone git@github.com:dpthorngren/Starlord.git
    cd Starlord
    pip install .

Example
--------------------

.. code:: toml

    [model]
    mist.wise = [7.5, 0.02]

    [sampling]
    sampler = "emcee"

    [output]
    terminal = true
