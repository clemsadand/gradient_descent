import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
import re
from typing import Callable, Union, List, Tuple
import io

# Import the optimizer class from the provided code
class GradientDescentOptimizer:
    """A simple gradient descent optimizer using PyTorch's automatic differentiation."""
    
    def __init__(self, 
             learning_rate: float = 0.1, 
             epsilon: float = 1e-6,
             max_iterations: int = 1000,
             stopping_criterion: str = 'function_diff',
             verbose: bool = True):
    """
    Initialize the optimizer.
    
    Args:
        learning_rate: Step size for gradient descent
        epsilon: Convergence threshold
        max_iterations: Maximum number of iterations
        stopping_criterion: One of 'function_diff', 'param_diff', 'gradient_norm'
        verbose: Whether to print progress information
    """
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.max_iterations = max_iterations
    self.stopping_criterion = stopping_criterion
    self.verbose = verbose
    self.iterations = 0
    self.trajectory = []

def minimize(self, func: Callable, initial_params: Union[float, List[float], torch.Tensor]) -> Tuple[torch.Tensor, float, List[str]]:
    """
    Minimize a function using gradient descent with automatic differentiation.
    
    Args:
        func: Function to minimize
        initial_params: Initial parameters (can be a scalar, list, or tensor)
        
    Returns:
        Tuple of (optimal parameters, optimal function value, log output)
    """
    # Convert initial parameters to tensor and set requires_grad
    if isinstance(initial_params, (int, float)):
        x = torch.tensor([float(initial_params)], requires_grad=True)
    elif isinstance(initial_params, list):
        x = torch.tensor(initial_params, dtype=torch.float32, requires_grad=True)
    elif isinstance(initial_params, torch.Tensor):
        x = initial_params.clone().detach().requires_grad_(True)
    else:
        raise TypeError("initial_params must be a number, list, or tensor")
    
    self.iterations = 0
    self.trajectory = [x.clone().detach().numpy()]
    
    prev_x = x.clone().detach()
    
    try:
        # Try to evaluate function at initial point
        prev_f_val = func(prev_x).item()
    except Exception as e:
        return x.detach(), 0.0, [f"Error evaluating function: {str(e)}"]
    
    log_output = []
    if self.verbose:
        log_output.append(f"Iter {0}: Params {prev_x.detach().numpy()} Function value {prev_f_val:.6f}")
    
    while self.iterations < self.max_iterations:
        try:
            # Compute function value
            f_val = func(x)
            
            # Compute gradients
            f_val.backward(retain_graph=True)
            
            # Check if gradients are computed correctly
            if x.grad is None:
                log_output.append("Error: Gradients are None. The function may not be differentiable.")
                break
                
            with torch.no_grad():
                # Update parameters
                x_new = x - self.learning_rate * x.grad
                
                # Record trajectory
                self.trajectory.append(x_new.clone().numpy())
                
                # Check stopping criterion
                if self.stopping_criterion == 'function_diff':
                    new_f_val = func(x_new).item()
                    converged = abs(new_f_val - prev_f_val) < self.epsilon
                    if self.verbose:
                        log_output.append(f"Iter {self.iterations+1}: Params {x_new.numpy()} Function value {new_f_val:.6f}")
                    prev_f_val = new_f_val
                elif self.stopping_criterion == 'param_diff':
                    converged = torch.norm(x_new - prev_x) < self.epsilon
                    if self.verbose:
                        log_output.append(f"Iter {self.iterations+1}: Params {x_new.numpy()} Param diff {torch.norm(x_new - prev_x).item():.6f}")
                elif self.stopping_criterion == 'gradient_norm':
                    converged = torch.norm(x.grad) < self.epsilon
                    if self.verbose:
                        log_output.append(f"Iter {self.iterations+1}: Params {x_new.numpy()} Gradient norm {torch.norm(x.grad).item():.6f}")
                else:
                    raise ValueError("Invalid stopping criterion")
                
                # Update for next iteration
                prev_x = x.clone().detach()
                x.grad.zero_()
                x.data = x_new.data
                
                self.iterations += 1
                
                if converged:
                    break
        except Exception as e:
            log_output.append(f"Error during optimization iteration {self.iterations}: {str(e)}")
            break
    
    # Final function value
    try:
        final_f_val = func(x).item()
    except Exception as e:
        log_output.append(f"Error evaluating final function value: {str(e)}")
        final_f_val = prev_f_val  # Use last successfully computed value
    
    if self.verbose:
        log_output.append(f"\nTotal iterations: {self.iterations}")
        log_output.append(f"Optimal parameters: {x.detach().numpy()}")
        log_output.append(f"Optimal function value: {final_f_val:.6f}")
    
    return x.detach(), final_f_val, log_output

def plot_trajectory(self, func: Callable, bounds: List[float] = None, contour: bool = False):
    """
    Plot the optimization trajectory.
    
    Args:
        func: The objective function
        bounds: Plot bounds [x_min, x_max] for 1D or [x_min, x_max, y_min, y_max] for 2D
        contour: Whether to plot contour for 2D functions
    """
    trajectory = np.array(self.trajectory)
    dim = trajectory.shape[1]
    
    fig = plt.figure(figsize=(10, 6))
    
    if dim == 1:
        self._plot_1d_trajectory(func, trajectory, bounds, fig)
    elif dim == 2:
        self._plot_2d_trajectory(func, trajectory, bounds, contour, fig)
    else:
        plt.text(0.5, 0.5, f"Cannot visualize trajectory for {dim}-dimensional parameters", 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    return fig

def _plot_1d_trajectory(self, func, trajectory, bounds=None, fig=None):
    """Plot trajectory for 1D optimization"""
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    
    # Set bounds if not provided
    if bounds is None:
        min_x = min(trajectory.min() - 1, -5)
        max_x = max(trajectory.max() + 1, 5)
        bounds = [min_x, max_x]
    
    # Plot function
    x = np.linspace(bounds[0], bounds[1], 1000)
    try:
        y = [func(torch.tensor([float(xi)], requires_grad=False)).item() for xi in x]
        plt.plot(x, y, 'b-', label='Function')
        
        # Plot trajectory points
        f_vals = [func(torch.tensor([float(xi)], requires_grad=False)).item() for xi in trajectory]
        plt.plot(trajectory, f_vals, 'ro-', label='Optimization path')
        
        # Highlight start and end points
        plt.plot(trajectory[0], f_vals[0], 'g*', markersize=10, label='Start')
        plt.plot(trajectory[-1], f_vals[-1], 'y*', markersize=10, label='End')
    except Exception as e:
        plt.text(0.5, 0.5, f"Error plotting function: {str(e)}", 
                ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Optimization Trajectory')
    plt.legend()
    plt.grid(True)
    
    return fig

def _plot_2d_trajectory(self, func, trajectory, bounds=None, contour=True, fig=None):
    """Plot trajectory for 2D optimization"""
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
    
    # Set bounds if not provided
    if bounds is None:
        margin = 1.0
        min_x = trajectory[:, 0].min() - margin
        max_x = trajectory[:, 0].max() + margin
        min_y = trajectory[:, 1].min() - margin
        max_y = trajectory[:, 1].max() + margin
        bounds = [min_x, max_x, min_y, max_y]
    
    try:
        if contour:
            # Create mesh grid for contour plot
            x = np.linspace(bounds[0], bounds[1], 50)  # Reduced from 100 to 50 for performance
            y = np.linspace(bounds[2], bounds[3], 50)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Evaluate function at grid points
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = func(torch.tensor([float(X[i, j]), float(Y[i, j])], requires_grad=False)).item()
            
            # Plot contour
            plt.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.7)  # Reduced from 50 to 20 levels
            plt.colorbar(label='f(x, y)')
            plt.contour(X, Y, Z, 10, colors='k', alpha=0.3)
        
        # Plot trajectory
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', label='Optimization path')
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'g*', markersize=10, label='Start')
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'y*', markersize=10, label='End')
    except Exception as e:
        plt.text(0.5, 0.5, f"Error plotting function: {str(e)}", 
                ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Trajectory')
    plt.legend()
    plt.grid(True)
    
    return fig

# Helper functions for parsing symbolic expressions
def sympy_to_torch_function(expr_str, variables):
    """Convert a SymPy expression string to a PyTorch function."""
    try:
        # Parse the expression
        expr = parse_expr(expr_str, transformations=standard_transformations)
        
        # Create the lambda function based on number of variables
        if len(variables) == 1:
            # 1D function
            x = sp.symbols('x')
            def torch_func(params):
                if isinstance(params, torch.Tensor):
                    if params.dim() == 0:  # Scalar tensor
                        return float(expr.subs(x, params.item()))
                    else:  # Vector tensor
                        return float(expr.subs(x, params[0].item()))
                return float(expr.subs(x, params))
        elif len(variables) == 2:
            # 2D function
            x, y = sp.symbols('x y')
            def torch_func(params):
                if isinstance(params, torch.Tensor):
                    return float(expr.subs([(x, params[0].item()), (y, params[1].item())]))
                return float(expr.subs([(x, params[0]), (y, params[1])]))
        elif len(variables) == 3:
            # 3D function
            x, y, z = sp.symbols('x y z')
            def torch_func(params):
                if isinstance(params, torch.Tensor):
                    return float(expr.subs([(x, params[0].item()), 
                                           (y, params[1].item()),
                                           (z, params[2].item())]))
                return float(expr.subs([(x, params[0]), (y, params[1]), (z, params[2])]))
        else:
            raise ValueError(f"Unsupported number of variables: {len(variables)}")
        
        # Wrap the function to make it compatible with PyTorch's autograd
        def wrapped_func(params):
            if not isinstance(params, torch.Tensor):
                if isinstance(params, (list, np.ndarray)):
                    params = torch.tensor(params, dtype=torch.float32, requires_grad=True)
                else:
                    params = torch.tensor([float(params)], requires_grad=True)
            
            result = torch_func(params)
            return torch.tensor(result, requires_grad=True)
        
        return wrapped_func
    
    except Exception as e:
        st.error(f"Error parsing function: {str(e)}")
        return None

def detect_variables(expr_str):
    """Detect variables used in the expression."""
    # Look for x, y, z variables
    variables = []
    if re.search(r'\bx\b', expr_str):
        variables.append('x')
    if re.search(r'\by\b', expr_str):
        variables.append('y')
    if re.search(r'\bz\b', expr_str):
        variables.append('z')
    
    return variables

def validate_expression(expr_str, variables):
    """Validate that the expression can be parsed and contains only allowed variables."""
    try:
        expr = parse_expr(expr_str, transformations=standard_transformations)
        
        # Check that only allowed variables are used
        expr_symbols = {str(symbol) for symbol in expr.free_symbols}
        for symbol in expr_symbols:
            if symbol not in variables:
                return False, f"Unexpected variable '{symbol}'. Only {', '.join(variables)} are allowed."
        
        return True, "Expression is valid."
    except Exception as e:
        return False, f"Error parsing expression: {str(e)}"

def get_example_functions():
    """Return a dictionary of example functions for different dimensions."""
    return {
        "1D": {
            "Simple Quadratic": "x**2 - 4*x + 4",
            "Cubic Function": "x**3 - 6*x**2 + 11*x - 6",
            "Trigonometric": "sin(x) + 0.1*x**2"
        },
        "2D": {
            "Bowl Function": "x**2 + y**2",
            "Rosenbrock Function": "100*(y - x**2)**2 + (1 - x)**2",
            "Saddle Function": "x**2 - y**2"
        },
        "3D": {
            "Simple 3D Bowl": "x**2 + y**2 + z**2",
            "3D Function": "x**2 + 2*y**2 + 3*z**2 + 2*x*y - 3*y*z",
            "Spherical Function": "sin(sqrt(x**2 + y**2 + z**2))"
        }
    }

def main():
    st.set_page_config(
        page_title="Gradient Descent Optimizer",
        page_icon="📉",
        layout="wide"
    )
    
    st.title("📉 Gradient Descent Function Optimizer")
    st.markdown("""
    This app allows you to minimize mathematical functions using gradient descent optimization.
    Enter a function symbolically, set the optimization parameters, and visualize the optimization trajectory.
    """)
    
    # Sidebar for optimization parameters
    st.sidebar.header("Optimization Parameters")
    
    learning_rate = st.sidebar.slider(
        "Learning Rate", 
        min_value=0.001, 
        max_value=1.0, 
        value=0.1, 
        step=0.001,
        format="%.3f"
    )
    
    epsilon = st.sidebar.slider(
        "Convergence Threshold (ε)", 
        min_value=1e-10, 
        max_value=1e-2, 
        value=1e-6, 
        step=1e-7,
        format="%.7f"
    )
    
    max_iterations = st.sidebar.slider(
        "Maximum Iterations", 
        min_value=10, 
        max_value=10000, 
        value=1000
    )
    
    stopping_criterion = st.sidebar.selectbox(
        "Stopping Criterion",
        options=["function_diff", "param_diff", "gradient_norm"],
        format_func=lambda x: {
            "function_diff": "Function Value Change",
            "param_diff": "Parameter Change",
            "gradient_norm": "Gradient Norm"
        }[x]
    )
    
    # Main area for function input
    st.header("Function Definition")
    
    # Modes: Manual input or Example selection
    input_mode = st.radio(
        "Function Input Mode",
        options=["Manual Input", "Choose Example Function"]
    )
    
    # Get example functions
    example_functions = get_example_functions()
    
    if input_mode == "Choose Example Function":
        # Select dimension first
        dim_options = list(example_functions.keys())
        selected_dim = st.selectbox("Select Dimension", dim_options)
        
        # Then select function for that dimension
        func_options = list(example_functions[selected_dim].keys())
        selected_func_name = st.selectbox("Select Example Function", func_options)
        
        # Set the function expression
        func_expr = example_functions[selected_dim][selected_func_name]
        st.text_area("Function Expression", value=func_expr, height=100, key="function_expr")
        
        # Determine variables based on dimension
        if selected_dim == "1D":
            variables = ["x"]
        elif selected_dim == "2D":
            variables = ["x", "y"]
        else:  # 3D
            variables = ["x", "y", "z"]
    else:
        # Manual input
        func_expr = st.text_area(
            "Enter your function using variables x, y, z",
            value="x**2 - 4*x + 4",
            height=100,
            help="Use standard mathematical notation. For example: x**2 - 4*x + 4, sin(x) + cos(y), etc."
        )
        
        # Detect variables
        detected_vars = detect_variables(func_expr)
        if not detected_vars:
            st.warning("No variables (x, y, z) detected in your expression!")
            variables = ["x"]  # Default to 1D
        else:
            variables = detected_vars
            st.info(f"Detected variables: {', '.join(variables)}")
    
    # Validate the function
    is_valid, validation_msg = validate_expression(func_expr, ['x', 'y', 'z'])
    if not is_valid:
        st.error(validation_msg)
    
    # Initial parameters input
    st.header("Initial Parameters")
    
    # Initialize initial parameter inputs based on dimensions
    initial_params = []
    col1, col2, col3 = st.columns(3)
    
    if 'x' in variables:
        with col1:
            x_init = st.number_input("Initial x", value=1.0, step=0.1)
            initial_params.append(x_init)
    
    if 'y' in variables:
        with col2:
            y_init = st.number_input("Initial y", value=1.0, step=0.1)
            initial_params.append(y_init)
    
    if 'z' in variables:
        with col3:
            z_init = st.number_input("Initial z", value=1.0, step=0.1)
            initial_params.append(z_init)
    
    # Visualization settings
    st.header("Visualization Settings")
    
    # For 1D and 2D only
    if len(variables) <= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot bounds
            if len(variables) == 1:
                x_min = st.number_input("X Axis Min", value=-5.0, step=0.5)
                x_max = st.number_input("X Axis Max", value=5.0, step=0.5)
                plot_bounds = [x_min, x_max]
            else:  # 2D
                x_min = st.number_input("X Axis Min", value=-5.0, step=0.5)
                x_max = st.number_input("X Axis Max", value=5.0, step=0.5)
                y_min = st.number_input("Y Axis Min", value=-5.0, step=0.5)
                y_max = st.number_input("Y Axis Max", value=5.0, step=0.5)
                plot_bounds = [x_min, x_max, y_min, y_max]
        
        with col2:
            # Contour plot option for 2D
            if len(variables) == 2:
                use_contour = st.checkbox("Show Contour Plot", value=True)
            else:
                use_contour = False
    else:
        # For 3D, don't need visualization settings
        plot_bounds = None
        use_contour = False
    
    # Run optimization
    if st.button("Run Optimization", type="primary"):
        if is_valid:
            # Convert symbolic expression to torch function
            torch_func = sympy_to_torch_function(func_expr, variables)
            
            if torch_func is not None:
                # Create optimizer
                optimizer = GradientDescentOptimizer(
                    learning_rate=learning_rate,
                    epsilon=epsilon,
                    max_iterations=max_iterations,
                    stopping_criterion=stopping_criterion,
                    verbose=True
                )
                
                # Run minimization
                with st.spinner("Optimizing function..."):
                    x_opt, f_opt, log_output = optimizer.minimize(torch_func, initial_params)
                
                # Display results
                st.header("Optimization Results")
                
                # Show optimal parameters
                st.subheader("Optimal Parameters")
                
                opt_params = x_opt.numpy()
                cols = st.columns(len(variables))
                
                for i, (var, col) in enumerate(zip(variables, cols)):
                    with col:
                        st.metric(f"Optimal {var}", f"{opt_params[i]:.6f}")
                
                # Show optimal function value
                st.metric("Optimal Function Value", f"{f_opt:.6f}")
                
                # Show convergence info
                st.subheader("Convergence Information")
                st.text(f"Converged after {optimizer.iterations} iterations")
                
                # Plot trajectory
                if len(variables) <= 2:  # Only plot for 1D and 2D
                    st.subheader("Optimization Trajectory")
                    fig = optimizer.plot_trajectory(torch_func, bounds=plot_bounds, contour=use_contour)
                    st.pyplot(fig)
                
                # Display optimization log
                with st.expander("Optimization Log", expanded=False):
                    for line in log_output:
                        st.text(line)
            else:
                st.error("Failed to convert expression to function. Please check your syntax.")
        else:
            st.error("Please fix the function expression before running optimization.")
    
    # Add mathematical background information
    with st.expander("About Gradient Descent Optimization"):
        st.markdown("""
        ### Gradient Descent Optimization
        
        Gradient descent is an iterative optimization algorithm used to minimize a function by moving in the direction of steepest descent.
        
        The basic algorithm works as follows:
        1. Start with initial parameters θ₀
        2. Compute the gradient ∇f(θ) of the function at the current parameters
        3. Update parameters: θₙ₊₁ = θₙ - α·∇f(θₙ) where α is the learning rate
        4. Repeat until convergence criteria are met
        
        ### Stopping Criteria
        
        * **Function Value Change**: Stop when |f(θₙ₊₁) - f(θₙ)| < ε
        * **Parameter Change**: Stop when ||θₙ₊₁ - θₙ|| < ε
        * **Gradient Norm**: Stop when ||∇f(θₙ)|| < ε
        
        ### Tips for Success
        
        * Choose an appropriate learning rate - too large may cause divergence, too small may be slow
        * Start with different initial parameters if you're not getting good results
        * For complex functions, consider using more iterations or relaxing the convergence threshold
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Created with Streamlit and PyTorch")

if __name__ == "__main__":
    main()
