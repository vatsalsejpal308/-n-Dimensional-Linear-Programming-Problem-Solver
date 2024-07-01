# Two-Phase-Simplex-for-LPP
This project offers a concise implementation of the Two Phase Simplex method, a robust algorithm tailored for solving linear programming problems. It efficiently optimizes linear objective functions amidst linear equality and inequality constraints, especially beneficial for non-canonical formulations.

**Features:**

->Efficient Algorithm: Utilizes the Two Phase Simplex method to efficiently find the optimal solution to 
  linear programming problems.

->Support for Constraints: Handles both equality and inequality constraints with ease, providing flexibility  
  in problem formulation.

->User-Friendly Interface: Designed with a user-friendly interface, making it easy to input problem data and 
  interpret results.

->Scalable and Versatile: Suitable for a wide range of linear programming problems across various domains,   
  from resource allocation to production planning.

**Usage:**

->Input Data: Provide the objective function coefficients, constraint coefficients, and constraints type 
  (equality/inequality).

->Execute Algorithm: Run the Two Phase Simplex algorithm to find the optimal solution.

->Interpret Results: Review the solution to understand the optimal values of decision variables and the     
  objective function.

**Output Format:**

It reads input.txt and returns the following dictionary in the given order:

->Initial Tableau (initial_tableau)

->Final Tableau (final_tableau)

->Status of Solution (solution_status) : “optimal”/ “infeasible”/ “unbounded”.

->Optimal Solution Vector (optimal_solution): Your code should return the optimal solution 
  vector x*, which contains the values of the decision variables [x1, x2, ..., xn] that maximize or 
  minimize the objective function, depending on the problem statement. This vector will be 
  in a standardized format, such as a Numpy array.
  
->Optimal Value (optimal_value): Real values.
  


