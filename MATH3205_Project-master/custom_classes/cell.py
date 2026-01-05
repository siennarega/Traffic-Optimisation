from dataclasses import dataclass, field
import numpy as np

@dataclass
class Cell:

    #type of cell: ordinary, merge, diverge, intersection
    type: str
    
    #Maximum number of cars flowing through cell at each each time #Q
    flow_lim: float = field(default_factory=float) 
    
    #Maximum number of cars residing in cell at each each time #N
    capacity: float = field(default_factory=float)
    
    #Ratio between shock-wave and free-flow speed of cell at each time #W
    wave_ratio: float = field(default_factory=float)

    #Initial demand in the cell by scenario
    init_demand: list = field(default_factory=list)

    #Demand by scenario and time-step (only set for origin cells)
    demand: np.ndarray | None = None

    #Turning ratio by scenario and output cell (only set for diverge cells)
    turn_ratio: np.ndarray | None = None

    #Cells sending outflow to this cell #P
    inputs: list = field(default_factory=list)
    
    #Cells recieving inflow from this cell #D
    outputs: list = field(default_factory=list)


    def __repr__(self):
        return f"{self.type} cell"
    
    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return id(self) == id(other)