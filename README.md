# ArtificialLife - Differentiable MPM (Material Point Method) Simulation with Taichi

This project implements a **Differentiable Material Point Method (MPM)** simulation using the Taichi programming language. The simulation is designed to optimize the shape and actuation of soft robots or deformable objects by minimizing a loss function that incorporates metrics such as distance traveled, stability, deformation, and shape penalties.

## Key Features

- **Differentiable Simulation**: The simulation is fully differentiable, allowing gradient-based optimization of shape parameters and actuation patterns.
- **Soft Robot Optimization**: The code optimizes the shape and actuation of a soft robot to achieve specific goals, such as moving toward a target or maintaining stability.
- **Customizable Shapes**: The initial shape of the robot can be customized using command-line arguments.
- **Actuation Patterns**: The robot is actuated using sinusoidal waves, and the actuation parameters (weights and biases) are optimized during the simulation.
- **Visualization**: The simulation results are visualized using Taichi's GUI, and frames are saved as images for further analysis or animation.

---

## How It Works

### Simulation Overview
1. **Material Point Method (MPM)**:
   - The simulation uses the MPM to model the behavior of particles on a grid.
   - Particles represent the material of the robot, and their positions, velocities, and deformation gradients are updated at each time step.

2. **Differentiable Physics**:
   - The simulation is differentiable, meaning gradients of the loss function with respect to the shape parameters and actuation patterns can be computed.
   - This allows for gradient-based optimization of the robot's shape and actuation.

3. **Loss Function**:
   - The loss function incorporates several metrics:
     - **Distance Traveled**: The robot should move as far as possible.
     - **Stability**: The robot should remain stable during motion.
     - **Deformation**: The robot should minimize deformation.
     - **Shape Penalty**: The robot's shape should stay within desired bounds (e.g., width and height constraints).

4. **Actuation**:
   - The robot is actuated using sinusoidal waves, with parameters (weights and biases) optimized during the simulation.
   - The actuation strength and frequency can be adjusted.

5. **Shape Optimization**:
   - The shape of the robot is defined by parameters such as position (`x`, `y`), width, and height.
   - These parameters are optimized to minimize the loss function.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Taichi (`pip install taichi`)
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
- `noise` library (`pip install noise`)

### Running the Code
1. Clone the repository or download the script.
2. Install the required dependencies.
3. Run the script with the desired command-line arguments.

Example:
```bash
python differentiable_mpm.py --iters 100 --shape 1
```

---

## Command-Line Arguments

| Argument       | Description                                                                 | Default Value |
|----------------|-----------------------------------------------------------------------------|---------------|
| `--iters`      | Number of optimization iterations.                                          | `100`         |
| `--shape`      | Initial shape of the robot. Choose from `1`, `2`, or `3`.                   | `3`           |

---

## Code Structure

### Key Components
1. **Fields and Parameters**:
   - `x`, `v`: Particle positions and velocities.
   - `C`, `F`: Deformation gradient and affine momentum.
   - `grid_v_in`, `grid_m_in`: Grid velocities and masses.
   - `actuation`: Actuation patterns for each actuator.

2. **Loss Function**:
   - The loss function is computed based on distance traveled, stability, deformation, and shape penalties.

3. **Optimization**:
   - Gradient descent is used to optimize the shape parameters and actuation patterns.

4. **Visualization**:
   - The simulation results are visualized using Taichi's GUI, and frames are saved as images.

### Functions
- `initialize_x()`: Initializes particle positions.
- `initialize_actuator_id()`: Initializes actuator IDs for particles.
- `p2g()`: Transfers particle data to the grid.
- `g2p()`: Transfers grid data back to particles.
- `compute_actuation()`: Computes actuation patterns.
- `compute_loss()`: Computes the loss function.
- `optimize_iteration()`: Performs one iteration of optimization.

---

## Example Usage

### Running the Simulation
To run the simulation with 100 optimization iterations and the default shape:
```bash
python differentiable_mpm.py --iters 100
```

To run the simulation with a different initial shape (e.g., shape 1):
```bash
python differentiable_mpm.py --iters 100 --shape 1
```

### Visualizing Results
- The simulation frames are saved in the `diffmpm/iterXXX/` folder.
- You can use these frames to create an animation or analyze the results.

---

## Customization

### Changing the Target
To change the target position, modify the `target` variable in the code:
```python
target = [0.8, 0.2]  # Target position (x, y)
```

### Adjusting Actuation Parameters
To adjust the actuation frequency or strength, modify the following variables:
```python
actuation_omega = 30  # Frequency of actuation
act_strength = 20     # Strength of actuation
```

### Adding New Shapes
To add a new initial shape, define a new set of shape parameters in the `main()` function:
```python
elif options.shape == 4:
    shape_params[0] = [0.0, 0.1, 0.7, 0.4]  # Base shape
    shape_params[1] = [0.0, 0.02, 0.2, 0.4]  # Actuator 1
    shape_params[2] = [0.3, 0.05, 0.05, 0.4]  # Actuator 2
    shape_params[3] = [0.355, 0.05, 0.05, 0.3]  # Actuator 3
```

---

## Output

- **Loss Plot**: A plot of the loss over optimization iterations is displayed at the end of the simulation.
- **Frames**: Simulation frames are saved in the `diffmpm/iterXXX/` folder.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- This project is inspired by the work on differentiable physics and soft robotics.
- Used the difftaichi git repo diffmpm.py as a base to start off of 

---

## Contact

For questions or feedback, please open an issue on GitHub or contact the author directly 
