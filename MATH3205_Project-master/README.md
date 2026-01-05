In order to run solution, navigate to main.py, and simply press run.

This will firstly initialise the optimiser, run the full formulation which extracts network data, constructs variables, objectives and constraints, and dump the results into a logger JSON file, titled, log_out.json. The lighting scheme from the full formulation run is then extracted, and re-formatted into a dictionary titled greens, and is fed into the simulator. The simulator will solve the second stage of the formulation under the specified deterministic lighting scheme, and results will also be dumped into file sim_log_out.json. Solution will then be visualised, and post-run cleanup occurs.

Different networks can be optimised by changing the file name on line 9. 
