import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation 

def main():
    parser = argparse.ArgumentParser(description="Race Car Simulation with dynamic obstacles")
    parser.add_argument("--params",default='params/eta_2s/scenario3_ttc.json' , type=str, help="Path to the JSON configuration file")
    parser.add_argument("--show", action="store_true", help="Show the plot after running the simulation")
    parser.add_argument("--save", action="store_true", help="Save the plot after running the simulation")
    parser.add_argument("--seed", default = np.random.randint(0, 1000), type=int, help="Random seed for setting the scenario")
    args = parser.parse_args()

    # Load parameters from the specified JSON file
    params_file = args.params
    SHOW = args.show
    SAVE = args.save
    seed = args.seed

    try:
        with open(params_file, "r") as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file {params_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Configuration file {params_file} is not a valid JSON file.")
        return

    # Run the simulation with the given parameters
    sim = Simulation(SAVE=True, params_file=params_file, seed=seed)
    sim.run()
    sim.plot_results()
    if SHOW:
        plt.show()
    if SAVE:
        sim.save_params()

if __name__ == "__main__":
    main()