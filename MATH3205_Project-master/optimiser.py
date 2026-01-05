from custom_classes.network import Network
from utils.visualiser import visualise_traffic
import gurobipy as gp
import numpy as np
import time
import json

class Optimiser:
        def __init__(self, network_json_file, K=1, T=60, Ncy=2, alpha=0.2):
                """
                Initialise the optimiser with network data and parameters.
                
                Arguments:
                network_json_file: str, path to the network JSON file
                K: int, number of scenarios (default 1)
                T: int, total number of time steps (default 60)
                Ncy: int, total number of cycles (default 2)
                alpha: float, weighting parameter for objective function (default 0.2)
                """

                self.nw = Network() #defining network
                self.nw.parse_json(network_json_file) #parsing network from json file
                self.K = K #number of scenarios 
                self.T = T  #total number of time steps
                self.Ncy = Ncy #total number of cycles
                self.alpha = alpha #weighting parameter for objective function
                self.nw.initialise_data_random(K, T) #initialise network data with random demand and turning ratios
                self.model = gp.Model("Traffic Optimisation") #defining gurobi model
                self.model.Params.Threads = 16 #setting number of threads for gurobi

                #model setup
                start = time.perf_counter() #time tracking
                self._extract_network_data() #extracting network data for optimisation
                print(f"_extract_network_data: {time.perf_counter() - start:.4f}s")

        def clean(self):
                """
                Clean up Gurobi model to free resources.
                """
                self.model.dispose()
                del self.model

                
        def run_full_formulation(self):
                """
                Run the full optimisation formulation which constructs decision 
                variables, constraints and objective.
                """
                start = time.perf_counter() #time tracking
                self._create_decision_variables() #creating decision variables
                print(f"_create_decision_variables: {time.perf_counter() - start:.4f}s")

                start = time.perf_counter() #time tracking
                self._create_constraints_and_objective() #creating constraints and objective
                print(f"_create_constraints_and_objective: {time.perf_counter() - start:.4f}s")

                self.model.optimize()

        def log_dump(self, k=0):
                """
                Dumps the optimisation results to a JSON file for further analysis.

                Arguments:
                k, scenarios: int
                """
                data = {}
                for int_id, intersection in self.R.items():
                        #initialize intersection data
                        data[int_id] = {}
                        data[int_id]["intersection"] = {}
                        data[int_id]["merge"] = {}
                        data[int_id]["diverge"] = {}
                        for key1, int_dict in intersection.int_cells.items():
                                for key2, cell in int_dict.items():
                                        #create a unique key for the cell based on its position
                                        key = f"({key1}, {key2})"
                                        
                                        #extract y and n values over time for scenario k
                                        y_vals = [self.y[cell, t, k].x for t in range(self.T)]
                                        n_vals = [self.n[cell, t, k].x for t in range(self.T)]  

                                        #store the data
                                        data[int_id]["intersection"][key] = {
                                                "outputs": [str(c) for c in cell.outputs],
                                                "inputs": [str(c) for c in cell.inputs],
                                                "y": y_vals,
                                                "n": n_vals
                                        }
                        #extract diverge cell data
                        for key, cell in intersection.div_cells.items():
                                y_vals = [self.y[cell, t, k].x for t in range(self.T)]
                                n_vals = [self.n[cell, t, k].x for t in range(self.T)]
                                
                                #store the data
                                data[int_id]["diverge"][key] = {
                                        "outputs": [str(c) for c in cell.outputs],
                                        "inputs": [str(c) for c in cell.inputs],
                                        "y": y_vals,
                                        "n": n_vals
                                }
                        #extract merge cell data
                        for key, cell in intersection.merge_cells.items():
                                y_vals = [self.y[cell, t, k].x for t in range(self.T)]
                                n_vals = [self.n[cell, t, k].x for t in range(self.T)]

                                #store the data
                                data[int_id]["merge"][key] = {
                                        "outputs": [str(c) for c in cell.outputs],
                                        "inputs": [str(c) for c in cell.inputs],
                                        "y": y_vals,
                                        "n": n_vals
                                }

                data[int_id]["destination"] = {}
                for cell in self.nw.destination_cells:
                        y_vals = [self.y[cell, t, k].x for t in range(self.T)]
                        n_vals = [self.n[cell, t, k].x for t in range(self.T)]

                        data[int_id]["destination"][id(cell)] = {
                                "outputs": [str(c) for c in cell.outputs],
                                "inputs": [str(c) for c in cell.inputs],
                                "y": y_vals,
                                "n": n_vals
                        }

                #origin demand sum
                data["origin demand sum"] = sum(cell.demand[k, t] for t in range(self.T) for cell in self.nw.origin_cells)
                data["flow obj contrib"] = sum((self.T - t) * self.y[cell,t,k].x for cell in self.nw.cells for t in range(self.T))
                data["destination obj contrib"] = sum(self.n[cell,t,k].x for cell in self.nw.destination_cells for t in range(self.T))

                #write to json file
                with open("log_out.json", "w") as f:
                        json.dump(data, f, indent=2)
        
        def get_lighting_scheme(self):
                """
                Returns the lighting scheme decision variables for the formulation.

                Returns: z1, z2
                """
                return self.z1, self.z2

        def _extract_network_data(self, print_data = False):
                """
                Extracts relevant network data for optimisation. Contains a flag for printing
                data.

                Arguments:
                print_data: bool (default False)
                """
                #Network Data
                nw = self.nw #network object
                self.C = nw.cells #all cells
                self.O = nw.origin_cells #origin cells
                self.D = nw.destination_cells #destination cells
                self.I = nw.intersection_cells #intersection cells
                self.M = nw.merge_cells #merge cells
                self.V = nw.diverge_cells #diverge cells
                self.R = nw.intersections #dictionary of intersections
                self.Ni = len(self.R) #number of intersections
                self.F = {intersection.id: intersection.phases for intersection in self.R.values()} #intersection phases with id
                self.Gmin = {
                (g_min): intersection.green_min
                for g_min, intersection in self.R.items()} #minimum green time per intersection 
                self.Gmax = {
                (g_max): intersection.green_max
                for g_max, intersection in self.R.items()} #maximum green time per intersection

                #Flag for printing data
                if print_data == True:
                        #Visualise Data
                        print("----------------Cells---------------------")
                        print("C is", self.C)
                        print("------------------------------------------")
                        print("O is", self.O)
                        print("------------------------------------------")
                        print("D is", self.D)
                        print("------------------------------------------")
                        print("R is", self.R)
                        print("------------------------------------------")
                        print("Ni is", self.Ni)
                        print("------------------------------------------")
                        print("I is", self.I)
                        print("------------------------------------------")
                        print("M is", self.M)
                        print("------------------------------------------")
                        print("V is", self.V)
                        print("------------------------------------------")
                        print("R is", self.R)
                        print("------------------------------------------")
                        print("Ni is", self.Ni)
                        print("-----------Intersection Properties------------")
                        print("F is", self.F)
                        print("------------------------------------------")
                        print("Gmin is", self.Gmin)
                        print("------------------------------------------")
                        print("Gmax is", self.Gmax)
                        print(len(self.C))


        def _create_decision_variables(self):
                """
                Create decision variables for optimisation.
                """
                
                model = self.model #gurobi model
                R = self.R #intersections
                F = self.F #intersection phases
                C = self.C #cells
                T = self.T #time steps
                K = self.K #scenarios
                Ncy = self.Ncy #cycles

                #Decision Variables

                #cycle length of intersection i
                self.l = {i: model.addVar() for i in R}

                #offset of intersection i
                self.o = {i: model.addVar() for i in R}

                #green length of intersection i at phase p
                self.g = {(i,j): model.addVar() for i in R for j in F[i]}

                #beginning time of green phase j in cycle Ncy of intersection i
                self.b = {(i,j,m): model.addVar(ub=T) for i in R for j in F[i] for m in range(Ncy)}

                #ending time of green phase j in cycle Ncy of intersection i
                self.e = {(i,j,m): model.addVar(ub=T) for i in R for j in F[i] for m in range(Ncy)}

                #binary variables linking b and e for time t,t+1 for T
                #z1 = 1 if time t is within the green phase j in cycle Ncy
                self.z1 = {(i,j,m,t): model.addVar(vtype=gp.GRB.BINARY) for i in R for j in F[i] for m in range(Ncy) for t in range(T)}
                #z2 = 1 if time t+1 is within the green phase j in cycle Ncy
                self.z2 = {(i,j,m,t): model.addVar(vtype=gp.GRB.BINARY) for i in R for j in F[i] for m in range(Ncy) for t in range(T)}

                #number of vehicles leaving cell c at time t,t+1 for T in scenario k
                self.y = {(c,t,k): model.addVar() for c in C for t in range(T) for k in range(K)}

                #number of vehicles in cell c at time t,t+1 for T in scenario k
                self.n = {(c,t,k): model.addVar() for c in C for t in range(T) for k in range(K)}


        def _create_constraints_and_objective(self):
                """
                Create constraints and objective for optimisation.
                """
                model = self.model
                R = self.R
                F = self.F
                Gmin = self.Gmin
                Gmax = self.Gmax
                C = self.C
                D = self.D
                I = self.I
                V = self.V
                O = self.O
                T = self.T
                K = self.K
                Ncy = self.Ncy
                alpha = self.alpha

                l = self.l
                o = self.o
                g = self.g
                b = self.b
                e = self.e
                z1 = self.z1
                z2 = self.z2
                y = self.y
                n = self.n

                #U = large number
                U = T+1
                #EPS = small number
                EPS = 10e-02

                #Constraints

                #1a
                #Enforces the relationship between time steps, t, and the start time, b, and end time,
                #e, of the green interval, ensuring that z1 is on when the current time step is greater than or
                #equal to the beginning time of the interval and off otherwise.
                oneA = {(i, j, m, t): (
                        model.addConstr(-U * z1[i,j,m,t] <= b[i,j,m] - t),
                        model.addConstr(b[i,j,m] - t <= U * (1 - z1[i,j,m,t]) - EPS))
                        for i in R for j in F[i] for m in range(Ncy) for t in range(T)}


                #1b
                #Enforces the relationship between time steps, t, and the start time, b, and end time,
                #e, of the green interval, ensuring that z2 is on when the current time step is smaller than or
                #equal to the end time of the interval and off otherwise.
                oneB = {(i,j,m,t): (
                        model.addConstr(-U*z2[i,j,m,t] + EPS <= t-e[i,j,m]),
                        model.addConstr(t-e[i,j,m] <= U*(1-z2[i,j,m,t])))
                        for i in R for j in F[i] for m in range(Ncy) for t in range(T)}

                #1c
                #Ensures that at each time step, there is a single phase with a green light.
                oneC = {(i,m,t):
                        model.addConstr(gp.quicksum(z1[i,j,m,t] + z2[i,j,m,t] for j in F[i]) <= 1 + len(F[i]))
                        for i in R for m in range(Ncy) for t in range(T)}

                #1d
                #Indicates that the offset of an intersection is smaller than the length of the cycle.
                oneD = {i:
                        model.addConstr(o[i]<=l[i])
                        for i in R}

                #1e
                #Allows for the variable, b, to link the level and offset variables across cycles.
                oneE = {(i,m):
                        model.addConstr(b[i,F[i][0],m]==l[i]*m-o[i]) 
                        for i in R for m in range(Ncy)}

                #1f
                #Enforces the ending interval time e as the sum of the beginning time b and green interval time g
                oneF = {(i,j,m):
                        model.addConstr(e[i,j,m]==b[i,j,m]+g[i,j])
                        for i in R for j in F[i] for m in range(Ncy)}

                #1g
                #Ensures that the value of the ending time of the previous stage equals the start of
                #the next, allowing for flow across phases.
                oneG = {(i,j,m):
                        model.addConstr(b[i,j,m]==e[i,F[i][F[i].index(j)-1],m])
                        for i in R for j in F[i] if j != F[i][0] for m in range(Ncy)}

                #1h
                #Allows for the sum of all green times across all phases to be equal to the cycle length.
                oneH = {i:
                        model.addConstr(l[i]==gp.quicksum(g[i,j] for j in F[i]))
                        for i in R}

                #1I
                #Bounds the green interval time between its specified minimum.
                oneI = {(i,j):(
                        model.addConstr(Gmin[i] <= g[i,j]),
                        model.addConstr(g[i,j]<= Gmax[i]))
                        for i in R for j in F[i]}

                #2A
                # minimise the negative value of the expected vehicle throughput of the
                #network over all time-steps with incentive to clear vehicles early.

                # Set the objective using the scalar weight
                model.setObjective(
                        gp.quicksum( (1/K) * ( 
                                gp.quicksum(
                                -n[c, t, k] for c in D for t in range(T)) - 
                        alpha * gp.quicksum(
                                (T - t) * y[c, t, k] for c in C for t in range(T)))
                        for k in range(K)
                ),
                gp.GRB.MINIMIZE)


                #2B
                #Ensures the number of vehicles leaving a cell is limited by the total vehicles within
                #that cell.
                twoB1 = {(c,t,k):
                        model.addConstr(y[c,t,k]<= n[c,t,k])
                        for c in C if c not in D for t in range(T) for k in range(K)}
                
                twoB2 = {(c,t,k):
                        model.addConstr(y[c,t,k]== n[c,t,k])
                        for c in D for t in range(T) for k in range(K)}

                #2C
                #Allows for the number of vehicles leaving a cell to be restricted by the cell flow
                #capacity.
                twoC = {(c,t,k):
                        model.addConstr(y[c,t,k] <= c.flow_lim)
                        for c in C if c not in D and c not in I.values() for t in range(T) for k in range(K)}
                
                #2D
                #Allows for intersection cells to restrict their capacity by related traffic signals (e.g.,
                #if the signal is red, capacity is zero).
                twoD = {(c,i,j,t,k):
                        model.addConstr(y[c,t,k]<=gp.quicksum((z1[i,j,m,t]+z2[i,j,m,t] -1)*(c.flow_lim) for m in range(Ncy)))
                        for i in R 
                        for j in F[i] 
                        for t in range(T) 
                        for k in range(K) 
                        for (int, src, _),c in I.items() if int == i and src==j}
                        

                #2E
                #Enforce that the number of vehicles leaving a cell is limited by the capacity of its
                #processing cell.
                twoE = {(c,cc,t,k):
                        model.addConstr(y[c,t,k]<=cc.flow_lim)
                        for c in C if c not in V.values() 
                        for cc in c.outputs 
                        for t in range(T) 
                        for k in range(K)}

                #2F
                #Also enforces that the number of vehicles leaving a diverge cell is limited by its
                #turning ratio since it has more than one processing cell.
                twoF = {(c,i,j,cc,t,k):
                        model.addConstr(y[c,t,k]<=cc.flow_lim / (c.turn_ratio[k, idx]))
                        for i in R 
                        for j in F[i] 
                        for c in V.values() 
                        for idx, cc in enumerate(c.outputs) if cc in I.values() # This relies on phasing simplification
                        for t in range(T) 
                        for k in range(K)}

                #2G
                #Limits the number of vehicles leaving a cell to the number of vehicles that can enter
                #its processing cells.
                twoG = {(c,cc,t,k):
                        model.addConstr(y[c,t,k]<=(cc.wave_ratio)*(cc.capacity-n[cc,t,k]))
                        for c in C if c not in V.values() 
                        for cc in c.outputs 
                        for t in range(T) 
                        for k in range(K)}

                #twoH
                #limits the number of vehicles leaving a diverge cell to the number of vehicles
                #that can enter its processing cells.
                twoH = {(c,t,k):
                        model.addConstr(y[c,t,k]<=(cc.wave_ratio)*(cc.capacity-n[cc,t,k])/(c.turn_ratio[k, idx]))
                        for c in V.values() 
                        for idx, cc in enumerate(c.outputs) 
                        for t in range(T) 
                        for k in range(K)}

                #twoI
                #Enforces flow conservation for all cells except O and I such the number of vehicles
                #in a cell between two consecutive time steps equal to the number of vehicle coming from
                #preceding flow (inflow) minus the number of vehicles leaving the cell (outflow).
                twoI = {(c,t,k):
                        model.addConstr(n[c,t+1,k] == n[c,t,k] + gp.quicksum(y[cc,t,k] for cc in c.inputs) - y[c,t,k])
                        for c in C if c not in O and c not in I.values() 
                        for t in range(T-1) 
                        for k in range(K)}

                #twoJ
                #Enforces flow conservation for O cells, where the number of vehicles in a cell between
                #consecutive time steps equals the inflow minus the outflow.
                twoJ = {(c,t,k):
                        model.addConstr(n[c,t+1,k] == n[c,t,k] + c.demand[k][t] - y[c,t,k])
                        for c in O 
                        for t in range(T-1) 
                        for k in range(K)}
        

                #twoK 
                #Enfoces flow conservation for I cells, where the number of vehicles in a cell between
                #consecutive time steps equals inflow minus outflow.
                twoK = {(c,t,k):
                        model.addConstr(n[c,t+1,k] == n[c,t,k] + c.turn_ratio[k]*y[c.inputs[0],t,k] - y[c,t,k])
                        for c in I.values() 
                        for t in range(T-1) 
                        for k in range(K)}
                
                #twoL
                #Allows the initial number of vehicles inside a cell to be set.
                twoL = {(c,k):
                        model.addConstr(n[c,0,k]==c.init_demand[k])
                        for c in C for k in range(K)}

                #twoM - 1
                #The amount of vehicles leaving a cell is non-negative.
                twoM1 = {(c,t,k):
                        model.addConstr(y[c,t,k]>=0)
                        for c in C for t in range(T) for k in range(K)}
                
                #twoM - 2
                #The amount of vehicles inside a cell is non-negative.
                twoM2 = {(c,t,k):
                        model.addConstr(n[c,t,k]>=0)
                        for c in C for t in range(T) for k in range(K)}
        


        def _benders_masterproblem(self):
                """
                Create and solve the Benders Master Problem.
                """

                #Master Problem
                BMP = gp.Model("Master Problem")

                #Master Problem Data
                R = self.R #intersections
                F = self.F #phases
                Ncy = self.Ncy #cycles
                T = self.T #time steps
                Gmin = self.Gmin #minimum green times
                Gmax = self.Gmax #maximum green times
                K = self.K #scenarios
                C = self.C #cells
                U = 10000000 #U = large number
                EPS = 1e-6 #EPS = small number

                #Master Problem Variables

                #cycle length of intersection i
                l = {i: BMP.addVar() for i in self.R} 

                #offset of intersection i
                o = {i: BMP.addVar() for i in self.R} 

                #green length of intersection i at phase j
                g = {(i,j): BMP.addVar() for i in self.R for j in self.F[i]}

                #beginning time of green phase j in cycle Ncy of intersection i
                b = {(i,j,m): BMP.addVar() for i in self.R for j in self.F[i] for m in range(self.Ncy)}
                
                #ending time of green phase j in cycle Ncy of intersection i
                e = {(i,j,m): BMP.addVar() for i in self.R for j in self.F[i] for m in range(self.Ncy)}

                #binary variable linking b and e for time t,t+1 for T
                z1 = {(i,j,m,t): BMP.addVar(vtype=gp.GRB.BINARY) for i in self.R for j in self.F[i] for m in range(self.Ncy) for t in range(self.T)}
                
                #binary variable linking b and e for time t,t+1 for T
                z2 = {(i,j,m,t): BMP.addVar(vtype=gp.GRB.BINARY) for i in self.R for j in self.F[i] for m in range(self.Ncy) for t in range(self.T)}
                
                #cost variable for scenario k
                theta = {k: BMP.addVar() for k in range(self.K)}

                #Master Problem Constraints 
                #1a
                #Enforces the relationship between time steps, t, and the start time, b, and end time,
                #e, of the green interval, ensuring that z1 is on when the current time step is greater than or
                #equal to the beginning time of the interval and off otherwise.
                oneA = {(i, j, m, t): (
                        BMP.addConstr(-U * z1[i,j,m,t] <= b[i,j,m] - t),
                        BMP.addConstr(b[i,j,m] - t <= U * (1 - z1[i,j,m,t]) - EPS))
                        for i in R for j in F[i] for m in range(Ncy) for t in range(T)}


                #1b
                #Enforces the relationship between time steps, t, and the start time, b, and end time,
                #e, of the green interval, ensuring that z2 is on when the current time step is smaller than or
                #equal to the end time of the interval and off otherwise
                oneB = {(i,j,m,t): (
                        BMP.addConstr(-U*z2[i,j,m,t] + EPS <= t-e[i,j,m]),
                        BMP.addConstr(t-e[i,j,m] <= U*(1-z2[i,j,m,t])))
                        for i in R for j in F[i] for m in range(Ncy) for t in range(T)}

                #1c
                #Ensures that at each time step, there is a single phase with a green light.
                oneC = {(i,m,t):
                        BMP.addConstr(gp.quicksum(z1[i,j,m,t] + z2[i,j,m,t] for j in F[i]) <= 1 + len(F[i]))
                        for i in R for m in range(Ncy) for t in range(T)}

                #1d
                #Indicates that the offset of an intersection is smaller than the length of the cycle.
                oneD = {i:
                        BMP.addConstr(o[i]<=l[i])
                        for i in R}

                #1e
                #Allows for the variable, b, to link the level and offset variables across cycles.
                oneE = {(i,m):
                        BMP.addConstr(b[i,F[i][0],m]==l[i]*m-o[i]) 
                        for i in R for m in range(Ncy)}

                #1f
                #Enforces the ending interval time e as the sum of the beginning time b and green interval time g
                oneF = {(i,j,m):
                        BMP.addConstr(e[i,j,m]==b[i,j,m]+g[i,j])
                        for i in R for j in F[i] for m in range(Ncy)}

                #1g
                #Ensures that the value of the ending time of the previous stage equals the start of
                #the next, allowing for flow across phases.
                oneG = {(i,j,m):
                        BMP.addConstr(b[i,j,m]==e[i,F[i][F[i].index(j)-1],m])
                        for i in R for j in F[i] if j != F[i][0] for m in range(Ncy)}

                #1h
                #Allows for the sum of all green times across all phases to be equal to the cycle length.
                oneH = {i:
                        BMP.addConstr(l[i]==gp.quicksum(g[i,j] for j in F[i]))
                        for i in R}

                #1I
                #Bounds the green interval time between its specified minimum.
                oneI = {(i,j):(
                        BMP.addConstr(Gmin[i] <= g[i,j]),
                        BMP.addConstr(g[i,j]<= Gmax[i]))
                        for i in R for j in F[i]}
                
                BMP.optimize()

                return BMP, l, o, g, b, e, z1, z2, theta
        
        
        def visualise_full_formulation_solution(self, k=0):
                """
                Visualises the solution of the full formulation.

                Arguments:
                k, scenario: int

                Returns: greens"""

                #data
                R = len(self.R.keys())
                D = 4
                T = self.T

                #solution arrays
                flows_in = np.zeros((R,D,T))
                flows_out = np.zeros((R,D,T))
                greens = np.zeros((R,D,T))
                
                #direction to index mapping
                dir_to_idx = {"N":0, "E":1, "S":2, "W":3}
                
                #extract solution
                for idx, intersection in self.R.items():
                        for phase in intersection.phases:
                                for t in range(T):
                                        light_is_on = False
                                        for m in range(self.Ncy):
                                                if round(self.z1[idx, phase, m, t].x + self.z2[idx, phase, m, t].x) == 2:
                                                        light_is_on = True
                                                        break
                                        greens[idx-1,dir_to_idx[phase],t] = int(light_is_on)
                
                num_ints = len(self.R.keys())

                if num_ints == 2:
                        positions = [(0.3, 0.7), (0.7, 0.7)]
                elif num_ints == 3:
                        positions = [(0.2, 0.7), (0.5, 0.7), (0.8, 0.7)]
                elif num_ints == 4:
                        positions = [(0.3, 0.7), (0.7, 0.7), (0.3, 0.3), (0.7, 0.3)]
                elif num_ints == 5:
                        positions = [
                                (0.5, 0.6),   # center
                                (0.5, 0.8),  # top
                                (0.7, 0.6),  # right
                                (0.5, 0.4),  # bottom
                                (0.3, 0.6)  # left
                                ]
                elif num_ints == 6:
                        positions = [
                                (0.25, 0.7), (0.5, 0.7), (0.75, 0.7),
                                (0.25, 0.4), (0.5, 0.4), (0.75, 0.4)
                                ]
                elif num_ints == 9:
                        positions = [
                                (0.25, 0.75), (0.5, 0.75), (0.75, 0.75),
                                (0.25, 0.5),  (0.5, 0.5),  (0.75, 0.5),
                                (0.25, 0.25), (0.5, 0.25), (0.75, 0.25)
                                ]
                else:
                        raise ValueError("Unexected number of intersections. ")

                #extract flows
                for idx, intersection in self.R.items():
                        for dir, div_cell in intersection.div_cells.items():
                                for t in range(T):
                                        flows_in[idx-1, dir_to_idx[dir], t] = self.y[div_cell, t, k].x

                        for dir, merge_cell in intersection.merge_cells.items():
                                for t in range(T):
                                        flows_out[idx-1, dir_to_idx[dir], t] = self.y[merge_cell, t, k].x
                #visualise traffic
                visualise_traffic(flows_in, flows_out, greens, positions)

                return greens