from optimiser import Optimiser
from Simulator import Simulator

def main():
        """
        Main function to run the optimisation and simulation.
        """
        #initialize optimiser - default as simple.json
        optimiser = Optimiser("3-int-line.json")
        T = optimiser.T
        K = optimiser.K
        Ncy = optimiser.Ncy
        network = optimiser.nw

        #Run optimisation
        optimiser.run_full_formulation()
        optimiser.log_dump()
        z1, z2 = optimiser.get_lighting_scheme()

        greens = {}
        dir_to_idx = {"N":0, "E":1, "S":2, "W":3}

        for idx, intersection in network.intersections.items():
                for phase in intersection.phases:
                        for t in range(T):
                                light_is_on = False
                                for m in range(Ncy):
                                        if round(z1[idx, phase, m, t].x + z2[idx, phase, m, t].x) == 2:
                                                light_is_on = True
                                                break
                                greens[idx,dir_to_idx[phase],t] = int(light_is_on)

        #Run simulation with extracted lighting scheme
        sim = Simulator(lighting_timing=greens, network=network, K=K, T=T)
        sim.simulate(alpha = 0.2)
        sim.log_dump()
        optimiser.visualise_full_formulation_solution()
        optimiser.clean()

if __name__ == "__main__":
        main()




