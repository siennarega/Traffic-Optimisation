from dataclasses import dataclass, field

@dataclass
class Intersection:
    id: int
    green_min: float
    green_max: float
    phases: list 
    internal_lim: int
    
    #Associated diverge cells
    div_cells: dict = field(default_factory=dict) 

    #Associated intersection cells
    int_cells: dict = field(default_factory=dict) 

    #Associated merge cells
    merge_cells: dict = field(default_factory=dict) 