# # Optimal Control of the Poisson equation
# *Section author: Jørgen S. Dokken [(dokken@simula.no)](mailto:dokken@simula.no)*.
#
# Original implementation in [dolfin-adjoint](https://github.com/dolfin-adjoint/dolfin-adjoint) was by Simon W. Funke.

# This demo solves the *mother problem* of PDE-constrained optimization: the optimal control of the Possion equation.
# Physically, this problem can be interpreted as finding the best heating/cooling of a cooktop to achieve a desired temperature profile.

# This example introduces the basics of how to solve optimization problems with DOLFINx-adjoint.

# ## Problem definition
# Mathematically, the goal is to minimize the following tracking type functional:
#
# $$
# \min_{f \in Q} J(u) = \frac{1}{2} \int_{\Omega} (u - d)^2 ~\mathrm{d}x + \frac{\alpha}{2}\int_{\Omega} f^2~\mathrm{d} x
# $$
# 
# subject to the Poisson equation with Dirichlet boundary conditions:
#
# $$
# \begin{align}
# - \kappa \Delta u &= f  && \text{in } \Omega, \\
# u &= 0 && \text{on } \partial\Omega, \\
# a &\leq f \leq b &&
# \end{align}
# $$
