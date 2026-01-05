import json
import sys
import os
from .intersection import Intersection
from utils import CAR_LEN, WAVE_SPEED, TIME_RES
from .cell import Cell
import random
import numpy as np

class Network:
    def __init__(self, cell_len=0.02):
        """
        Initialise the network with empty data structures.
        
        Arguments:
        cell_len: float, length of each cell in the network (default 0.02 km)
        """
        #Initialise network data structures
        self.cells = {}
        self.ordinary_cells = []
        self.origin_cells = []
        self.destination_cells = []
        self.intersection_cells = {}
        self.merge_cells = {}
        self.diverge_cells = {}
        self.intersections = {}
        self.cell_len = cell_len

    def parse_json(self, filename):
        """
        Parse the network structure from a JSON file.
        
        Arguments:
        filename: str, path to the network JSON file
        """
        filename = os.path.join("synthetic_data", "networks", filename)

        #Load network data from JSON file
        with open(filename, "r") as f:
            data = json.load(f)

        for inter_data in data["intersections"]:

            # Extract intersection data
            inter_id = inter_data["id"]
            green_min, green_max = inter_data["signal_bounds"]
            internal_speed_lim = inter_data["internal_lim"]
            intersection_fn = os.path.join("synthetic_data", "intersections", inter_data["file_name"])

            wave_ratio = WAVE_SPEED / internal_speed_lim
            capacity = int(self.cell_len / CAR_LEN - 1)
            flow_lim = int(internal_speed_lim / 3600 / self.cell_len * TIME_RES)
            
            #Load intersection connection data
            with open(intersection_fn, "r") as f:
                connect_data = json.load(f)

            #Validate intersection data
            if connect_data["phases"] != len(inter_data["connections"]):
                print("ERROR: Specified phases in intersection JSON does not match " \
                "number of connection in network JSON. ")
                sys.exit(1)

            #Create intersection object
            intersection = Intersection(
                id=inter_id,
                green_min=green_min,
                green_max=green_max,
                phases=connect_data["link_keys"],
                internal_lim=internal_speed_lim
            )

            #create merge, diverge and intersection cells
            intersection.merge_cells = {dir : Cell(type="merge") for dir in connect_data["link_keys"]}
            intersection.div_cells = {dir : Cell(type="diverge") for dir in connect_data["link_keys"]}
            intersection.int_cells = {dir : {sink_dir : Cell(type="intersection") 
                                             for sink_dir in connect_data["link_keys"] if sink_dir != dir}
                                             for dir in connect_data["link_keys"]}
            
            #Links between diverge and intersection cells
            for dir, cell in intersection.div_cells.items():
                for sink in intersection.int_cells[dir].values():
                    cell.outputs.append(sink)
                    sink.inputs.append(cell)
                
                #Set diverge cell params
                self._set_cell_data(cell, wave_ratio, capacity, flow_lim)
            
            for dir, int_dict in intersection.int_cells.items():
                for sink, int_cell in int_dict.items():
                    int_cell.outputs.append(intersection.merge_cells[sink])
                    intersection.merge_cells[sink].inputs.append(int_cell)
                
                    #Set intersection cell params
                    self._set_cell_data(int_cell, wave_ratio, capacity, flow_lim)

            for dir, cell in intersection.merge_cells.items():
                #Set merge cell params
                self._set_cell_data(cell, wave_ratio, capacity, flow_lim)

            self.intersections[inter_id] = intersection

        #Create intersection cells dictionary
        self.intersection_cells = {
            (intersection_id, outer, inner): cell
            for intersection_id, intersection in self.intersections.items()
            for outer, inner_dict in intersection.int_cells.items()
            for inner, cell in inner_dict.items()} #intersection cells
        
        #create merge cells dictionary
        self.merge_cells = {
            (intersection_id, direction): cell
            for intersection_id, intersection in self.intersections.items()
            for direction, cell in intersection.merge_cells.items()} #merge cells

        #create diverge cells dictionary
        self.diverge_cells = {
            (intersection_id, direction): cell
            for intersection_id, intersection in self.intersections.items()
            for direction, cell in intersection.div_cells.items()} #diverge cells
        
        #Create ordinary cells and link connections between intersections
        for inter_data in data["intersections"]:
            source_int = inter_data["id"]
            for conn in inter_data["connections"]:
                sink_int = conn["connection id"]
                conn_dist = conn["distance"]
                source_dir = conn["direction"]
                lanes = conn["lanes"]
                speed_lim = conn["speed_lim"]

                wave_ratio = WAVE_SPEED / speed_lim
                capacity = int(self.cell_len / CAR_LEN - 1) * lanes
                flow_lim = int(speed_lim / 3600 / self.cell_len * TIME_RES * lanes)


                if sink_int == 0:
                    
                    #Create origin and destination cells
                    dest = Cell(type="destination")
                    origin = Cell(type="origin")

                    self._set_cell_data(origin, wave_ratio, capacity, flow_lim)
                    self._set_cell_data(dest, wave_ratio, float('inf'), float('inf'))

                    self.origin_cells.append(origin)
                    self.destination_cells.append(dest)
                    
                    #Link origin and destination to intersection cells
                    try:
                        source = self.intersections[source_int].merge_cells[source_dir]
                        sink = self.intersections[source_int].div_cells[source_dir]
                    except Exception as e:
                        print(e)
                        print("ERROR: Invalid direction given for origin / destination connection.")
                        sys.exit(1)

                    #Link origin to diverge cells
                    cur_cell = Cell(type="ordinary")
                    origin.outputs.append(cur_cell)
                    cur_cell.inputs.append(origin)
                    self.ordinary_cells.append(cur_cell)
                    self._set_cell_data(cur_cell, wave_ratio, capacity, flow_lim)

                    for k in range(int(conn_dist / self.cell_len) + 1):
                        
                        #Create ordinary cells between origin and merge cell
                        next_cell = Cell(type="ordinary")
                        cur_cell.outputs.append(next_cell)
                        next_cell.inputs.append(cur_cell)
                        cur_cell = next_cell
                        self.ordinary_cells.append(cur_cell)
                        self._set_cell_data(cur_cell, wave_ratio, capacity, flow_lim)
                    
                    cur_cell.outputs.append(sink)
                    sink.inputs.append(cur_cell)

                    #Link merge cell to destination cells
                    cur_cell = Cell(type="ordinary")
                    source.outputs.append(cur_cell)
                    cur_cell.inputs.append(source)
                    self.ordinary_cells.append(cur_cell)
                    self._set_cell_data(cur_cell, wave_ratio, capacity, flow_lim)

                    for k in range(int(conn_dist / self.cell_len) + 1):
                        
                        #Create ordinary cells between merge cell and destination
                        next_cell = Cell(type="ordinary")
                        cur_cell.outputs.append(next_cell)
                        next_cell.inputs.append(cur_cell)
                        cur_cell = next_cell
                        self.ordinary_cells.append(cur_cell)
                        self._set_cell_data(cur_cell, wave_ratio, capacity, flow_lim)
                    
                    cur_cell.outputs.append(dest)
                    dest.inputs.append(cur_cell)
                    
                else:

                    sink_dir = conn["connection dir"]

                    #Create a connection of ordinary cells between intersections
                    if sink_int not in self.intersections.keys():
                        print("ERROR: Unknown intersection ID specified in network connection.")
                        sys.exit(1)
                    
                    try:
                        source_cell = self.intersections[source_int].merge_cells[source_dir]
                    except Exception as e:
                        print("ERROR: Could not find merge cell for intersection connection.")
                        sys.exit(1)

                    if len(source_cell.outputs) != 0:
                        print("ERROR: Doubly connected output. JSON incorrectly formatted.")
                        sys.exit(1)
                    
                    cur_cell = Cell(type="ordinary")
                    source_cell.outputs.append(cur_cell)
                    cur_cell.inputs.append(source_cell)
                    self.ordinary_cells.append(cur_cell)
                    self._set_cell_data(cur_cell, wave_ratio, capacity, flow_lim)

                    for k in range(int(conn_dist / self.cell_len) + 1):
                        
                        #Create ordinary cells between intersections
                        next_cell = Cell(type="ordinary")
                        cur_cell.outputs.append(next_cell)
                        next_cell.inputs.append(cur_cell)
                        cur_cell = next_cell
                        self.ordinary_cells.append(cur_cell)
                        self._set_cell_data(cur_cell, wave_ratio, capacity, flow_lim)
                    
                    #Link to sink intersection diverge cell
                    try:
                        sink_cell = self.intersections[sink_int].div_cells[sink_dir]
                    except Exception as e:
                        print("ERROR: Could not find diverge cell for intersection connection.")
                        sys.exit(1)

                    if len(sink_cell.inputs) != 0:
                        print("ERROR: Doubly connected input. JSON incorrectly formatted.")
                        sys.exit(1)
                    
                    cur_cell.outputs.append(sink_cell)
                    sink_cell.inputs.append(cur_cell)
        
        self.cells = set(self.ordinary_cells) | set(self.destination_cells) | set(self.origin_cells) \
            | set(self.merge_cells.values()) | set(self.diverge_cells.values()) | set(self.intersection_cells.values()) 
        
    def _set_cell_data(self, cell, wave_ratio, capacity, flow_lim):
        """
        Set the parameters for a cell.
        
        Arguments:
        cell: Cell, the cell to set parameters for
        wave_ratio: float, ratio between shock-wave and free-flow speed
        capacity: float, maximum number of cars residing in the cell
        flow_lim: float, maximum number of cars flowing through the cell
        """
        cell.wave_ratio = 1.0
        cell.capacity = capacity * 4.0
        cell.flow_lim = flow_lim * 4.0

    def initialise_data_random(self, num_scenarios, num_timesteps, seed=29):
        """
        Initialise cell data (demand and turning ratios) randomly for testing.
        
        Arguments:
        num_scenarios: int, number of demand scenarios to generate
        num_timesteps: int, number of time-steps in each scenario
        seed: int, random seed (default 29)"""
        random.seed(seed)
        np.random.seed(seed)

        #Initialise demand and turning ratios for each cell
        for cell in sorted(self.cells, key=lambda c: id(c)):
            cell.init_demand = [random.randint(0, int(self.cell_len / CAR_LEN) - 1) for i in range(num_scenarios)]

            #Set turning ratios for diverge cells
            if cell in self.diverge_cells.values():
                if len(cell.outputs) == 3:
                    cell.turn_ratio = np.random.dirichlet((0.3, 0.5, 0.2), num_scenarios)
                    for idx, cc in enumerate(cell.outputs):
                        cc.turn_ratio = cell.turn_ratio[:, idx]
                elif len(cell.outputs) == 2:
                    cell.turn_ratio = np.random.dirichlet((0.5, 0.5), num_scenarios)
                    for idx, cc in enumerate(cell.outputs):
                        cc.turn_ratio = cell.turn_ratio[:, idx]
                else:
                    raise ValueError("Unexpected number of outputs in a diverge cell. ")

            #Set demand for origin cells
            if cell in self.origin_cells:
                cell.demand = np.round(np.random.uniform(0, min(cell.capacity, cell.flow_lim), (num_scenarios, num_timesteps)))