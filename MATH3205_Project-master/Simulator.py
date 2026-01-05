import json

class Simulator():
    def __init__(self, lighting_timing, network, K, T):
        """
        Initialise the simulator with lighting timing and network data.
        
        Arguments:
        lighting_timing: dict, mapping (intersection id, direction index, time) to 0/1
        network: Network object, representing the traffic network
        K: int, number of scenarios
        T: int, total number of time steps
        """

        self.lighting_timing = lighting_timing
        self.network = network
        self.k = K
        self.T = T
    
    def log_dump(self, k=0):
        """
        Dumps simulation results to a JSON file for scenario k.

        Arguments:
        k: int, scenario index to log (default 0)
        """
        #store data in dictionary
        data = {}
        for int_id, intersection in self.network.intersections.items():
                #initialize data
                data[int_id] = {}
                data[int_id]["intersection"] = {}
                data[int_id]["merge"] = {}
                data[int_id]["diverge"] = {}
                
                #Intersection cells
                for key1, int_dict in intersection.int_cells.items():
                        for key2, cell in int_dict.items():
                                #create unique key for cell
                                key = f"({key1}, {key2})"
                                
                                #get y and n values over time
                                y_vals = [self.y[cell, t, k] for t in range(self.T)]
                                n_vals = [self.n[cell, t, k] for t in range(self.T)]  

                                #store data
                                data[int_id]["intersection"][key] = {
                                        "outputs": [str(c) for c in cell.outputs],
                                        "inputs": [str(c) for c in cell.inputs],
                                        "y": y_vals,
                                        "n": n_vals
                                }

                #Diverge cells
                for key, cell in intersection.div_cells.items():
                        #get y and n values over time
                        y_vals = [self.y[cell, t, k] for t in range(self.T)]
                        n_vals = [self.n[cell, t, k] for t in range(self.T)]

                        #store data
                        data[int_id]["diverge"][key] = {
                                "outputs": [str(c) for c in cell.outputs],
                                "inputs": [str(c) for c in cell.inputs],
                                "y": y_vals,
                                "n": n_vals,
                        }

                #Merge cells
                for key, cell in intersection.merge_cells.items():
                        y_vals = [self.y[cell, t, k] for t in range(self.T)]
                        n_vals = [self.n[cell, t, k] for t in range(self.T)]

                        data[int_id]["merge"][key] = {
                                "outputs": [str(c) for c in cell.outputs],
                                "inputs": [str(c) for c in cell.inputs],
                                "y": y_vals,
                                "n": n_vals
                        }
        data[int_id]["destination"] = {}
        for cell in self.network.destination_cells:
            y_vals = [self.y[cell, t, k] for t in range(self.T)]
            n_vals = [self.n[cell, t, k] for t in range(self.T)]

            data[int_id]["destination"][id(cell)] = {
                    "outputs": [str(c) for c in cell.outputs],
                    "inputs": [str(c) for c in cell.inputs],
                    "y": y_vals,
                    "n": n_vals
            }
             
        #origin demand sum
        data["origin demand sum"] = sum(cell.demand[k, t] for t in range(self.T) for cell in self.network.origin_cells)
        data["flow obj contrib"] = sum((self.T - t) * self.y[cell,t,k] for cell in self.network.cells for t in range(self.T))
        data["destination obj contrib"] = sum(self.n[cell,t,k] for cell in self.network.destination_cells for t in range(self.T))

        #write to json
        with open("sim_log_out.json", "w") as f:
                json.dump(data, f, indent=2)

    
    def simulate(self, alpha):

        """
        Simulate traffic flow through the network over time for all scenarios given a lighting timing scheme.
        
        Arguments:
        alpha: float, weighting parameter for objective function
        
        Returns:
        y: dict, mapping (cell, time, scenario) to outflow values
        n: dict, mapping (cell, time, scenario) to number of cars in cell
        obj_vals: dict, mapping scenario index to objective value
        """

        #initialise n and y as dictionaries: (cell, t, k)
        self.y = {(cell, t, k): 0 for cell in self.network.cells for t in range(self.T) for k in range(self.k)}
        self.n = {(cell, t, k): 0 for cell in self.network.cells for t in range(self.T) for k in range(self.k)}

        #twoL
        #Allows the initial number of vehicles inside a cell to be set.
        for cell in self.network.cells:
                for k in range(self.k):
                    self.n[(cell, 0, k)] = cell.init_demand[k] #2L

        #direction to index mapping
        dir_to_idx = {"N":0, "E":1, "S":2, "W":3}


        #Main simulation
        for k in range(self.k):
            for t in range(self.T):
                for cell in self.network.cells:
    
                    y_constraints = []

                    #2B
                    # #Ensures the number of vehicles leaving a cell is limited by the total vehicles within
                    #that cell. 
                    y_constraints.append(self.n[(cell, t, k)])

                    #2C
                    #Allows for the number of vehicles leaving a cell to be restricted by the cell flow
                    #capacity.
                    if cell.type not in ["destination", "intersection"]:
                        y_constraints.append(cell.flow_lim)

                    #2D
                    #Allows for intersection cells to restrict their capacity by related traffic signals (e.g.,
                    #if the signal is red, capacity is zero).
                    if cell.type == "intersection":
                        for (int_id, src_phase, _), c in self.network.intersection_cells.items():
                            if c == cell:
                                if self.lighting_timing[int_id, dir_to_idx[src_phase],t]:
                                    y_constraints.append(c.flow_lim)
                                else:
                                    y_constraints.append(0.0)

                    #2E
                    #Enforce that the number of vehicles leaving a cell is limited by the capacity of its
                    #processing cell.
                    if cell.type != "diverge":
                        y_constraints += [up.flow_lim for up in cell.outputs]

                    #2F
                    #Also enforces that the number of vehicles leaving a diverge cell is limited by its
                    #turning ratio since it has more than one processing cell.
                    if cell.type == "diverge":
                        y_constraints += [up.flow_lim/cell.turn_ratio[k, idx] for idx, up in enumerate(cell.outputs)]

                    #2G
                    #Limits the number of vehicles leaving a cell to the number of vehicles that can enter
                    #its processing cells.
                    if cell.type != "diverge":
                        y_constraints += [up.wave_ratio*(up.capacity-self.n[(up, t, k)]) for up in cell.outputs]

                    #2H
                    #limits the number of vehicles leaving a diverge cell to the number of vehicles
                    #that can enter its processing cells.
                    if cell.type == "diverge":
                        y_constraints += [(cc.wave_ratio*(cc.capacity-self.n[cc, t, k]))/cell.turn_ratio[k, idx] for idx, cc in enumerate(cell.outputs)]

                    #choose feasible y values
                    y_val = max(0, min(y_constraints))
                    self.y[(cell, t, k)] = y_val
                
                for cell in self.network.cells:
                
                    n_constraints = []

                    #2I
                    #Enforces flow conservation for all cells except O and I such the number of vehicles
                    #in a cell between two consecutive time steps equal to the number of vehicle coming from
                    #preceding flow (inflow) minus the number of vehicles leaving the cell (outflow).
                    if cell.type not in ["origin", "intersection"]:
                        n_constraints += [self.n[(cell, t, k)] + sum(self.y[(up, t, k)] for up in cell.inputs) - self.y[(cell, t, k)]]
        
                    #2J
                    #Enforces flow conservation for O cells, where the number of vehicles in a cell between
                    #consecutive time steps equals the inflow minus the outflow.
                    elif cell.type == "origin":
                        n_constraints += [self.n[(cell, t, k)] + cell.demand[k][t] - self.y[(cell, t, k)]]

                    #2K
                    #Enfoces flow conservation for I cells, where the number of vehicles in a cell between
                    #consecutive time steps equals inflow minus outflow.
                    elif cell.type == "intersection":
                         n_constraints += [self.n[(cell, t, k)] + cell.turn_ratio[k]*self.y[cell.inputs[0], t, k] - self.y[(cell, t, k)]]
                    else:
                        print(f"Unexpected cell type detected {cell.type}. Simulation error. ")

                    #choose feasible n values
                    n_val = max(0, min(n_constraints))
                    self.n[(cell, t+1, k)] = n_val
                 
        obj_vals = {}
        for k in range(self.k):
            #calculate and store objective value for scenario k
            obj_k = sum(-self.n[(cell, t, k)] for cell in self.network.destination_cells for t in range(self.T)) \
                    - alpha * sum((self.T - t) * self.y[(cell, t, k)] for cell in self.network.cells for t in range(self.T)) 
            obj_vals[k] = obj_k

        print(f"Simulation complete. Subproblem objectives per scenario: {obj_vals}")
        return self.n, self.y, obj_vals
                



    

