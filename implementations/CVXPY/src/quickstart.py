import cvxpy as cp

# Problem dimensions
n, m = 2, 3

# Decision variable (what we solve for)
x = cp.Variable(n)

# Parameters (inputs that change at runtime)
A = cp.Parameter((m, n))
b = cp.Parameter(m)

# Build the problem
problem = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ x - b)),
    [x >= 0]
)

# Check if the problem is DCP-compliant
assert problem.is_dcp(), "Problem is not DCP-compliant"


# Create the Layer
import torch
from cvxpylayers.torch import CvxpyLayer

layer = CvxpyLayer(
    problem,
    parameters=[A, b],  # CVXPY parameters (in order)
    variables=[x]       # Variables to return
)

# Solve & Differentiate

# Create tensors with gradients enabled
A_t = torch.randn(m, n, requires_grad=True)
b_t = torch.randn(m, requires_grad=True)

# Forward: solve the optimization
(solution,) = layer(A_t, b_t)

# Backward: compute gradients
loss = solution.sum()
loss.backward()

print(f"Solution: {solution}")
print(f"dL/dA: {A_t.grad}")
print(f"dL/db: {b_t.grad}")