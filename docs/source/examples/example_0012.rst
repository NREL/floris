example_0012_optimize_yaw_wind_rose_parallel.py
===============================================

The code for this example can be found here: `example_0012_optimize_yaw_wind_rose_parallel.py 
<https://github.com/NREL/floris/blob/develop/examples/example_0012_optimize_yaw_wind_rose_parallel.py>`_

This example performs the same yaw optimization for an example wind farm for a series of wind speed and direction combinations comprising a wind rose as :doc:`Example 0011 <example_0011>`, but using parallel computing with the :py:class:`YawOptimizationWindRoseParallel<floris.tools.optimization.YawOptimizationWindRoseParallel>` class in the :py:mod:`optimization<floris.tools.optimization>` module. The YawOptimizationWindRoseParallel class uses the MPIPoolExecutor method of the mpi4py.futures module to parallelize the baseline power and optimization operations for different wind speed and wind direction combinations. All plots and results should be identical to those from :doc:`Example 0011 <example_0011>`.  

This example is designed to be run on NREL's Eagle HPC system, but could be modified to run on other HPC systems. To run the example on Eagle, the following command can be used:

::

    $ sbatch runscript_example_0012

The runscript file `runscript_example_0012 <https://github.com/NREL/floris/blob/develop/examples/hpc_utilities/example_0012/runscript_example_0012>`_, located in the ``examples/hpc_utilities/example_0012/`` directory, sets the number of parallel tasks to 36, sets a time limit of 4 hours, initializes the floris environment and sets some environment variables using the `Floris_Python` command, then runs the example using the mpi4py.futures module. Note that the project handle, email address, and path to the floris examples directory need to be set. 

::

    #!/bin/bash
    #SBATCH --ntasks=36                             # Request number of CPU cores
    #SBATCH --time=4:00:00                       # Job should run for time
    #SBATCH --account=<project_handle>                 # Accounting
    #SBATCH --job-name=yaw_opt_example            # Job name
    #SBATCH --mail-user <your email address>    # user email for notifcations
    #SBATCH --mail-type ALL                         # ALL will notify for BEIGN,END,FAIL
    #SBATCH --output=yaw_opt_example.%j.out      # %j will be replaced with job ID

    source $HOME/.bash_profile
    source $HOME/.bashrc
    Floris_Python
    cd <floris/examples path>
    srun python -m mpi4py.futures example_0012_optimize_yaw_wind_rose_parallel.py

Before submitting `runscript_example_0012 <https://github.com/NREL/floris/blob/develop/examples/hpc_utilities/example_0012/runscript_example_0012>`_, a conda environment called "floris" should be created, 
containing the floris package, and the following command should be added
to ``~/.bashrc``:

::

    Floris_Python()
    {
       module purge
       module load conda/5.3
       module load intel-mpi/2018.0.3
       export PREFIX=~/.conda-envs/floris
       export PATH=$PREFIX/bin:$PATH
       export FI_PROVIDER_PATH=$PREFIX/lib/libfabric/prov
       export LD_LIBRARY_PATH=$PREFIX/lib/libfabric:$PREFIX/lib/release_mt:$LD_LIBRARY_PATH
       source activate floris
       export I_MPI_PMI_LIBRARY=/nopt/slurm/current/lib/libpmi.so
    }

In `example_0012_optimize_yaw_wind_rose_parallel.py 
<https://github.com/NREL/floris/blob/develop/examples/example_0012_optimize_yaw_wind_rose_parallel.py>`_, first, the wind farm coordinates and some optimization parameters are specified and the initial setup is computed for an example wind farm. Although wind direction and yaw uncertainty are not included in this example by default, by setting include_unc to True, wind direction uncdertainty is included, as explained in :doc:`Example 0011a <example_0011a>`.

::

    # Define wind farm coordinates and layout
    wf_coordinate = [39.8283, -98.5795]

    # set min and max yaw offsets for optimization
    min_yaw = 0.0
    max_yaw = 25.0

    # Define minimum and maximum wind speed for optimizing power. 
    # Below minimum wind speed, assumes power is zero.
    # Above maximum_ws, assume optimal yaw offsets are 0 degrees
    minimum_ws = 3.0
    maximum_ws = 15.0

    # Instantiate the FLORIS object
    fi = wfct.floris_interface.FlorisInterface("example_input.json")

    # Set wind farm to N_row x N_row grid with constant spacing 
    # (2 x 2 grid, 5 D spacing)
    D = fi.floris.farm.turbines[0].rotor_diameter
    N_row = 2
    spc = 5
    layout_x = []
    layout_y = []
    for i in range(N_row):
        for k in range(N_row):
            layout_x.append(i*spc*D)
            layout_y.append(k*spc*D)
    N_turb = len(layout_x)

    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y),wind_direction=270.0,wind_speed=8.0)
    fi.calculate_wake()

    # option to include uncertainty
    include_unc = False
    unc_options={'std_wd': 4.95, 'std_yaw': 0.0,'pmf_res': 1.0, 'pdf_cutoff': 0.95}

Similar to :doc:`Example 0011 <example_0011>`, but using the :py:class:`YawOptimizationWindRoseParallel<floris.tools.optimization.YawOptimizationWindRoseParallel>` class, the baseline power and optimized power are found for each wind speed and wind direction combination from the wind rose by creating an instance of the :py:class:`YawOptimizationWindRoseParallel<floris.tools.optimization.YawOptimizationWindRoseParallel>` class. The :py:meth:`calc_baseline_power()
<floris.tools.optimization.YawOptimizationWindRoseParallel.calc_baseline_power>` method is used to find the wind farm power and individual turbine power values for each wind direction and wind speed for baseline and no-wake scenarios. Next, the :py:meth:`optimize()
<floris.tools.optimization.YawOptimizationWindRoseParallel.optimize>` method is used to find the optimal wind farm power, individual turbine power values, and optimal yaw offsets for each wind speed and wind direction.

::

    # Instantiate the parallel optimization object
    yaw_opt = YawOptimizationWindRoseParallel(fi, df.wd, df.ws, 
                                   minimum_yaw_angle=min_yaw, 
                                   maximum_yaw_angle=max_yaw,
                                   minimum_ws=minimum_ws,
                                   maximum_ws=maximum_ws,
                                   include_unc=include_unc,
                                   unc_options=unc_options)

    # Determine baseline power
    df_base = yaw_opt.calc_baseline_power()

    # Perform optimization
    df_opt = yaw_opt.optimize()
