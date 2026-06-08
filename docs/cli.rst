Command Line Interface
======================

The full list of command line options can be displayed with ``starlord --help``.  This page organizes this information by function.

.. Admonition:: TLDR

    - Run the sampler: ``starlord myfile.toml -o myposterior.npz``
    - Run a forward model test case: ``starlord myfile.toml -dt "1,3,5"``
    - Analyze an input file: ``starlord -da myfile.toml``
    - View available grids ``starlord -g``
    - View a grid in detail ``starlord -g grid_name``
    - View a grid file: ``starlord some_grid.npz``
    - Analyze an output file: ``starlord posterior.npz``


Running Starlord on a Model File
--------------------------------
You can run the starlord with ``starlord myfile.toml``, which will read the input file, compile the model if needed, and create and run a sampler.  You can override some of the options specified in the model file:

-s SET_CONST, --set-const SET_CONST
    Set a model constant. For example, ``c.observed_j_mag`` in your model file would be set with ``-s obsered_j_mag=12.5``. This overrides any values set in the model file. You can set multiple constants in this way by repeating the argument.

-o [FILENAME], --output [FILENAME]
    Sets the file to write outputs to, overriding the ``output.file`` option in the model.

--corner-plot CORNER_PLOT
    Specify a file to write a corner plot to, if ``corner.py`` is installed.  This option is not supported for batch runs.


Batch Runs
^^^^^^^^^^
You can run a model repeatedly across a range of constant values specified in a csv file.  This is similar to running starlord repeatedly with ``-s`` to set the constants, except that it avoids reloading the grids each time, supports running each row (set of constants) in parallel, and can produce an output csv summarizing the results.  These may be specified with the following options:

-b BATCH, --batch BATCH
    Run for a range of constants, pulled from the given csv file.

--batch-summary BATCH_SUMMARY
    File to write batch run summary information to as a csv.

--batch-threads BATCH_THREADS
    Set the number of threads to run in parallel during a batch run.

The csv file you provide with batch should have one column per constant used in the model file, with an identical name; extraneous columns will be ignored.  If the column ``name`` is given, this will be used to name each of the output files and/or rows in the output summary file, otherwise the row index will be used.

As an example, if you used the constants ``c.wise_w1_obs``, ``c.wise_w1_err``, ``c.parallax_obs``, and ``c.parallax_err`` in your model file, the csv batch file might look like (this data is fictitious):

.. code::

   name, wise_w1_obs, wise_w1_err, parallax_obs, parallax_err
   HD23634, 13.534, 0.021, 15.3, 0.043
   HD670342, 23.635, 0.053, 23.162, 0.0836
   HD928725, 9.634, 0.026, 10.742, 0.0263

You could then sample the posterior for each case simultaneously with ``starlord mymodel.toml -b myconsts.csv --batch-summary summary_out.csv --batch-threads 3``.

Analyzing Model Files
---------------------
As with most programming, you are unlikely to write your starlord model file perfectly on the first try.  These analysis options are provided to help you find and resolve errors.  Using the analysis info and test case features is strongly recommended -- an example is shown in :doc:`quickstart/stars`.

-d, --dry-run
    Exit without running the sampler, usually in combination with one of the subsequent options.

-a, --analyze, --analyse
    Print analysis info for the model, including the parameters and variables used, the likelihood and prior terms, and a list of grids used.

-t TEST_CASE, --test-case TEST_CASE
    Tests the forward model and likelihood at the parameters given by TEST_CASE, which are the model parameters separated by commas (no spaces).  If a parameter is negative, you need to enclose TEST_CASE in quotation marks to avoid confusing the CLI parser.

--dep-graph
    Render the deferred variable dependencies with graphviz, if it is installed.  This is mainly useful for debugging new grids whose inputs depend on other grid outputs.

-c, --code
    Print the generated Cython code; this is probably only useful if you are familiar with Cython.

Examining Output Files
----------------------
Once you have run a model and saved the outputs, you can see the results again with ``starlord myposterior.npz``, which will print various metadata about the run and the same stats summary you saw if terminal output was enabled.  The only relevant option is ``--corner-plot myplot.png``, which outputs a corner plot exactly as if you had provided that argument when you ran the model in the first place.

If you wish to access the data from within Python, you may do so with ``numpy.load`` directly, using :func:`starlord.load_posterior` to get a dictionary of metadata and data outputs, or with :func:`starlord.load_to_frame` to load it as a Pandas DataFrame (so long as Pandas is installed).

Examining Available Grids
-------------------------
The list of grids available to Starlord can be printed with ``starlord -g``.  This will list the name of the grid, its input parameters, and a potentially abbreviated list of output and derived values.  The last input name is followed by an arrow ``->`` and the last output is followed by a semicolon ``;``.  For example, the Mist grids could be listed as:

.. code:: none

    Available grids:
        mist(logTeff, logG, feh, Av -> bc_2MASS_H, bc_2MASS_J, bc_2MASS_Ks, bc_ACS_HRC_F220W, +253; 2MASS_H, 2MASS_J, 2MASS_Ks, ACS_HRC_F220W, +261)
        mistInvAge(logMass0, feh0, logAge -> eep)
        mistTracks(logMass0, feh0, eep -> delta_nu, logAge, logMass, logRadius, +3; density, logG, logL)

From this you can see that the grid would allow you to use e.g. ``mist.2MASS_J`` as a constraint in your model, and that the inputs ``logTeff``, ``logG``, ``feh``, ``Av`` would be defined as a result. However, the full list was truncated rather than display the more than two hundred output variables.  To see the full grid, you must examine one grid in detail by naming it, as in ``starlord -g mist``.  For example:

.. code:: none

    Grid mist
        Input                       Min        Max     Length     Default Mapping
      0 logTeff                   3.398          6         70     d.mistTracks.logTeff--i
      1 logG                         -4        9.5         26     d.mistTracks.logG--i
      2 feh                          -4       0.75         18     p.feh
      3 Av                            0          6         13     p.Av
    Outputs
        Output                      Min        Max
      4 bc_2MASS_H               -15.75       3.68
      5 bc_2MASS_J               -16.24      2.802
      6 bc_2MASS_Ks              -15.48      3.935
      7 bc_ACS_HRC_F220W            -99     0.6799
      8 bc_ACS_HRC_F250W            -99     0.6371
      9 bc_ACS_HRC_F330W            -99     0.5389
    [...]
    518 WFPC2_F675W          -2.5 * d.mistTracks.logL--i + 9.74 - d.mist.bc_WFPC2_F675W--i - 5*(math.log10(d. ...
    519 WFPC2_F791W          -2.5 * d.mistTracks.logL--i + 9.74 - d.mist.bc_WFPC2_F791W--i - 5*(math.log10(d. ...
    520 WFPC2_F814W          -2.5 * d.mistTracks.logL--i + 9.74 - d.mist.bc_WFPC2_F814W--i - 5*(math.log10(d. ...
    521 WFPC2_F850LP         -2.5 * d.mistTracks.logL--i + 9.74 - d.mist.bc_WFPC2_F850LP--i - 5*(math.log10(d ...
    522 WISE_W1              -2.5 * d.mistTracks.logL--i + 9.74 - d.mist.bc_WISE_W1--i - 5*(math.log10(d.mist ...
    523 WISE_W2              -2.5 * d.mistTracks.logL--i + 9.74 - d.mist.bc_WISE_W2--i - 5*(math.log10(d.mist ...
    524 WISE_W3              -2.5 * d.mistTracks.logL--i + 9.74 - d.mist.bc_WISE_W3--i - 5*(math.log10(d.mist ...
    525 WISE_W4              -2.5 * d.mistTracks.logL--i + 9.74 - d.mist.bc_WISE_W4--i - 5*(math.log10(d.mist ...

In this case, the truncation [...] is in the documentation only; the actual output lists all 525 outputs and derived values.  For the inputs and outputs, the actual minimum and maximum values are shown so you can be sure your priors and likelihood terms fall within those bounds.  Inputs also list the length of the grid in that dimension and the default mapping; in this case we can see that the metallicity and extinction are set as parameters, whereas the effective temperature and gravity are calculated from the ``mistTracks`` grid.

.. note::

    The above instructions examine grids stored by starlord.  If you prefer to store them elsewhere, you can still get the grid summary with ``starlord path/to/mygrid.npz`` -- note that ``-g`` is now omitted.

Other Options
---------------
  -h, --help
    Show the basic CLI help message and exit, ignoring all other command line options.
  --version
    Show the program's version number and exit
  -p, --plain-text
    Prevents the use of ANSI escape sequences (bold text, colors, etc), for if your terminal doesn't support them (or you are piping output directly to a file).
  -v, --verbose
    Print extra debugging information, though this is mainly useful for debugging starlord rather than a given model file.
