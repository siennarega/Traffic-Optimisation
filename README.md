This repository contains development of the solution implemented by authors of European
of Journal Operations Research Article titled, Traffic Signal Control Under Stochastic Traffic
Demand and Vehicle Turning. Authors aim to solve the problem of traffic congestion by optimising traffic
signal control schedules at multiple intersections to maximise vehicle throughput on corridors
or road networks. This occurs under time-varying, stochastic demand originating at origins and stochastic turning ratios at intersections over a range of scenarios.

Included is development of synthetic intersection and network data implemented as JSON files, a data infrastructure solution containing classes to support optimisation, a simulator to assess the efficiency of traffic light schemes to replace the
solution sub-problem to enable future efficient implementation of logic-based Benders
Decomposition, along with visualisation of network with representation of intersection in/out-flow, and green light
timings.


In order to run solution, navigate to main.py, and simply press run. This will firstly initialise the optimiser, run the full formulation which extracts network data, constructs variables, objectives and constraints, and dump the results into a logger JSON file, titled, log_out.json. The lighting scheme from the full formulation run is then extracted, and re-formatted into a dictionary titled greens, and is fed into the simulator. The simulator will solve the second stage of the formulation under the specified deterministic lighting scheme, and results will also be dumped into file sim_log_out.json. Solution will then be visualised, and post-run cleanup occurs. Different networks can be optimised by changing the file name on line 9. 

This project is developed as part of UQ Course MATH3205, Further Topics in Operations Research, alongside teammate Marcus.
