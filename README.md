# NMPC-TTC 


## 📌 Description
This project simulates a miniature car navigating a track with dynamic obstacles. The simulation considers prediction horizons, reference velocities, and collision avoidance techniques to model realistic urban scenarios.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- ACADOS and CASADI installations for python :  [https://docs.acados.org/installation/](https://docs.acados.org/installation/index.html#linux-mac)
- Required dependencies (install with the command below)

## 📊 Visualizations
### Scenario (S1)

![Scenario (S1)](files/simulation_s1.gif)

![Scenario (S2)](files/simulation_s2.gif)

![Scenario (S3)](files/simulation_s3.gif)




## 🔧 Usage
To run the simulation, execute:
```bash
python main.py
```
Alternatively, to specify parameters, use:
```bash
python main.py --params params/scenario1.json
```

## ⚙️ Configuration
The simulation parameters are stored in JSON files inside the `params/` directory. Example:
```json
{
    "scenario": 1,
    "TRACK_FILE": "LMS_Track6.txt",
    "PREDICTION_HORIZON": 5.0,
    "TIME_STEP": 0.1,
    "MAX_SIMULATION_TIME": 20.0,
    "DIST_THRESHOLD": 1.0,
    "N_OBSTACLES_MAX": 3,
    "OBSTACLE_WIDTH": 0.15,
    "OBSTACLE_LENGTH": 0.25,
    "eta": 1.0
}
```
Feel free to test your own parameters and your own scenarios!

## 🌟 Features
- 🚀 Predictive control of a miniature car
- 🏎️ Dynamic obstacle avoidance 
- 📊 Plots and analyzes results
- 🔁 Supports multiple scenarios with JSON configuration and txt files

## 🤝 Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch`
3. Commit changes: `git commit -m "Added a new feature"`
4. Push: `git push origin feature-branch`
5. Open a Pull Request


## 🙌 Acknowledgments
- This project contains modified parts of the code that has been used for the simulations and experiments associated with the 
publication: *NMPC for Racing Using a Singularity-Free Path-Parametric Model with Obstacle Avoidance - Daniel Kloeser, Tobias Schoels, Tommaso Sartor, Andrea Zanelli, Gianluca Frison, Moritz Diehl. Proceedings of the 21th IFAC World Congress, Berlin, Germany - July 2020*. 

