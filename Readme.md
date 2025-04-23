# Gradient Descent Optimizer Streamlit App

This application provides an interactive interface for function minimization using gradient descent optimization. Users can input mathematical functions symbolically, set optimization parameters, and visualize the optimization trajectory.

## Features

- **Symbolic Function Input**: Enter mathematical functions using variables x, y, and z
- **Example Functions**: Choose from pre-defined example functions in 1D, 2D, and 3D
- **Customizable Optimization Parameters**: Adjust learning rate, convergence threshold, max iterations, and stopping criteria
- **Interactive Visualization**: View optimization trajectory with contour plots for 2D functions
- **Detailed Results**: Get optimal parameters, function value, and convergence information

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages: streamlit, torch, numpy, matplotlib, sympy

### Installation

1. Clone this repository or download the application files
2. Install the required packages:

```
pip install -r requirements.txt
```

### Running the App

```
streamlit run app.py
```

## Usage

1. **Define your function**:
   - Enter a mathematical expression using x, y, and z variables
   - Or choose from example functions

2. **Set initial parameters**:
   - Provide starting values for x, y, and z (depending on function dimension)

3. **Configure optimization settings**:
   - Adjust learning rate, convergence threshold, and maximum iterations
   - Select a stopping criterion

4. **Set visualization parameters** (for 1D and 2D functions):
   - Define plot bounds
   - Enable/disable contour plotting for 2D functions

5. **Run the optimization**:
   - Click "Run Optimization" to start the process
   - View results including optimal parameters, function value, and visualization

## Examples

### 1D Functions
- Simple Quadratic: `x**2 - 4*x + 4`
- Cubic Function: `x**3 - 6*x**2 + 11*x - 6`

### 2D Functions
- Bowl Function: `x**2 + y**2`
- Rosenbrock Function: `100*(y - x**2)**2 + (1 - x)**2`

### 3D Functions
- Simple 3D Bowl: `x**2 + y**2 + z**2`
- Spherical Function: `sin(sqrt(x**2 + y**2 + z**2))`

## How It Works

The application uses PyTorch's automatic differentiation capabilities to compute gradients for gradient descent optimization. The optimization algorithm follows these steps:

1. Parse the user's symbolic function using SymPy
2. Convert the function to a PyTorch-compatible function
3. Perform gradient descent optimization:
   - Compute function value and gradients at current parameters
   - Update parameters in the direction of steepest descent
   - Check convergence criteria
4. Visualize the optimization trajectory (for 1D and 2D functions)

## Acknowledgments

- This application is built on top of a custom gradient descent optimizer class
- Uses PyTorch for automatic differentiation
- Uses SymPy for symbolic mathematics
- Built with Streamlit for the web interface
