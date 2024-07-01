import numpy as np
import math

class SolveILP:
	def __init__(self, obj):
		'''obj = 1 for min and 0 for max'''
		self.obj = obj
		self.A = []  # will be converted to ndarray in standerdize
		self.b = []  # will be converted to ndarray in standerdize
		self.c = []  # will be converted to ndarray in standerdize
		self.constr = []  # <= if 1; = if 2; >= if 3
		self.varcount = 0
		self.tableau = np.empty([1, 1])  # will be converted to ndarray in createTableau
		self.status = "unbounded"
		self.Dualstatus = "unbounded"
		self.original_num_vars = 0
		self.num_vars = 0
		self.bfs = 0
		self.cuts = 0

	def standardize(self):
		if self.obj == 0:  # if maximization
			for i in range(len(self.c)):
				self.c[i] = -self.c[i]
		num_constraints = len(self.constr)
		num_vars = len(self.A[0])
		self.original_num_vars = num_vars
		self.bfs = np.zeros(self.original_num_vars)
		A_slack = np.zeros((num_constraints, num_vars + num_constraints))

		for i in range(num_constraints):
			if self.constr[i] == 1:
				A_slack[i, :num_vars] = self.A[i, :]
				A_slack[i, num_vars + i] = 1
			elif self.constr[i] == 2:
				A_slack[i, :num_vars] = self.A[i, :]
			elif self.constr[i] == 3:
				A_slack[i, :num_vars] = self.A[i, :]
				A_slack[i, num_vars + i] = -1
		i = num_vars
		while i<A_slack.shape[1]:
			if np.count_nonzero(A_slack[:, i]) == 0:
				A_slack = np.delete(A_slack, i, 1)
			else:
				i+=1
		c_slack = np.zeros(A_slack.shape[1])
		c_slack[:num_vars] = self.c
		self.A = A_slack
		self.c = c_slack
		for i in range(num_constraints):
			if self.b[i] < 0:
				for j in range(self.A.shape[1]):
					self.A[i][j] = -self.A[i][j]
				self.b[i] = -self.b[i]
		self.num_vars = self.c.shape[0]

	def createTableau(self):
		self.tableau.astype(float)
		m = self.tableau.shape[0] - 1
		while (self.tableau[0][1] < -1e-6):
			# check feasible
			opt, j = self.isOptimal()
			#added this condn below here @ वत्सल 
			if(opt):
				self.status = "infeasible"
				return
			c_j = self.tableau[0, j]
			l = -1
			ratio = 1e8
			for i in range(1, m + 1):
				if (self.tableau[i, j] > 1e-6):
					if (ratio > self.tableau[i, 1] / self.tableau[i, j]):
						ratio = self.tableau[i, 1] / self.tableau[i, j]
						l = i
			if (l < 0):
				self.status = "infeasible"
				return 1
			pivot_ele = self.tableau[l, j]
			# update basis index
			self.tableau[l, 0] = j - 2
			# update lth row
			self.tableau[l, 1:] = 1 / pivot_ele * self.tableau[l, 1:]
			# update rows except lth
			for i in range(m + 1):
				if i == l:
					continue
				self.tableau[i, 1:] -= self.tableau[l, 1:] * self.tableau[i, j]

		m, n = self.tableau.shape[0]-1,self.num_vars
		self.tableau = self.tableau[:,:n+2]
		i = 1
		while(i < self.tableau.shape[0]):
			if self.tableau[i,0] >= n:
				f = 1
				for j in range(2, n + 2):
					if (abs(self.tableau[i,j]) > 1e-6):
						f = 0
						l = i
						pivot_ele = self.tableau[l, j]
						# update basis index
						self.tableau[l, 0] = j - 2
						# update lth row
						self.tableau[l, 1:] = 1 / pivot_ele * self.tableau[l, 1:]
						# update rows except lth
						for k in range(self.tableau.shape[0]):
							if k == l:
								continue
							self.tableau[k, 1:] -= self.tableau[l, 1:] * self.tableau[k, j]
						i+=1
						break

				if f:
					self.tableau = np.delete(self.tableau, i, 0)
					self.A = np.delete(self.A, i-1, 0)
					self.b = np.delete(self.b, i-1, 0)
			else:
				i+=1

	def tableau_maker_phase1(self):
		m, n = self.A.shape
		cost_array = np.full(m, -1)
		reduced_costs = cost_array @ self.A
		# Phase 1: Add artificial variables
		A_phase1 = np.hstack((self.A, np.eye(m)))
		# m more variables
		array = np.zeros(m + 1)
		a = n
		for i in range(len(array)):
			if (i != 0):
				array[i] = a
				a += 1
		new_array = np.expand_dims(array, axis=1)

		reduced_cost = np.hstack((reduced_costs, np.zeros(m)))

		sum = 0
		for i in self.b:
			sum += i
		cost = - sum

		b_added = np.insert(self.b, 0, cost)
		b_reshaped = np.expand_dims(b_added, axis=1)
		tableau_with_costs = np.vstack((reduced_cost, A_phase1))

		tableau_with_costs = np.hstack((b_reshaped, tableau_with_costs))
		tableau_with_costs = np.hstack((new_array, tableau_with_costs))
		self.tableau = tableau_with_costs

	def isOptimal(self):
		for i in range(2, self.tableau.shape[1]):
			if (self.tableau[0, i] < -1e-6):
				return False, i
		self.status = "optimal"
		return True, 0

	def isDualOptimal(self):
		# return false and index of negative basic variable if not optimal. true, 0 otherwise
		for i in range(1, self.tableau.shape[0]):
			if (self.tableau[i, 1] < -1e-6):
				return False, i
		self.Dualstatus = "optimal"
		return True, 0

	def phase2TableauSolver(self):
		# assuming Basis coulmn i.e. 0th in tableau is 0 indexed
		self.tableau = self.tableau.astype(float)
		self.tableau[0, 1] = 0
		m = self.tableau.shape[0] - 1

		for i in range(1, m + 1):
			self.tableau[0, 1] -= self.c[int(self.tableau[i, 0])] * self.tableau[i, 1]
		A_B = np.zeros((m, m))  # basis matrix
		for i in range(m):
			for j in range(m):
				A_B[j, i] = self.A[j, int(self.tableau[i + 1, 0])]
		c_B = np.zeros(m)
		for i in range(m):
			c_B[i] = self.c[int(self.tableau[i + 1, 0])]
		c_B = np.transpose(c_B)
		# get c hat
		c_ = np.transpose(self.c) - np.dot(c_B, np.dot(np.linalg.inv(A_B), self.A))
		# update row 1
		for i in range(2, self.tableau.shape[1]):
			self.tableau[0, i] = c_[i - 2]
		# tableau ready
		# solve it
		it = 10000
		while (it):
			# check feasible
			opt, j = self.isOptimal()
			if (opt):
				return 0
			c_j = self.tableau[0, j]
			l = -1
			ratio = 1e8
			for i in range(1, m + 1):
				if (self.tableau[i, j] > 1e-6):
					if (ratio > self.tableau[i, 1] / self.tableau[i, j]):
						ratio = self.tableau[i, 1] / self.tableau[i, j]
						l = i
			if (l < 0):
				self.status = "unbounded"
				return 1
			pivot_ele = self.tableau[l, j]
			# update basis index
			self.tableau[l, 0] = j - 2
			# update lth row
			self.tableau[l, 1:] = 1 / pivot_ele * self.tableau[l, 1:]
			# update rows except lth
			for i in range(m + 1):
				if i == l:
					continue
				self.tableau[i, 1:] -= self.tableau[l, 1:] * self.tableau[i, j]
			it -= 1

	def isInteger(self):
		f = True
		idx = 0
		for i in range(1,self.tableau.shape[0]):
			x = self.tableau[i,1]
			if abs(round(x)-x) < 1e-6:
				self.tableau[i,1] = round(x)
			else:
				f = False
				idx = i
				return f, i
		return f, idx

	def solveRelaxed(self):
		self.standardize()
		self.tableau_maker_phase1()
		self.createTableau()

		if(self.status != "infeasible"):
			self.phase2TableauSolver()

	def addCuttingPlane(self,i):
		source_row = self.tableau[i].copy()
		fractional_parts = np.zeros(len(source_row))
		for j in range(2,len(source_row)):
			fractional_parts[j] = source_row[j] - math.floor(source_row[j]) # 1st and second element are name and its value
		gomory_cut_constraint = np.zeros(len(source_row) + 1) # last row which will be added in end
		gomory_cut_constraint[-1] = 1 # last element will be 1
		gomory_cut_constraint[0] = len(source_row) - 2 # new gomory variable name i.e len(source_row) - 2 is no.of variable in variable
		gomory_cut_constraint[1] = -(source_row[1] -math.floor(source_row[1])) # value of new variable
		basis_elem = self.tableau[:,0]
		for u in range(2,len(fractional_parts)):
			# if u-2 not in basis_elem: #at 2nd index x0 will be present and so on #redundant. basis elem wale colmn waise hi integer hote hai
			gomory_cut_constraint[u] = -fractional_parts[u]
			if(abs(gomory_cut_constraint[u]-round(gomory_cut_constraint[u])) < 1e-6):
				gomory_cut_constraint[u] = 0
		m,n = self.tableau.shape
		#adding 0 to end of all existing tableau row
		self.tableau = np.hstack((self.tableau, np.zeros((m, 1))))
		#PUSHING new formed constraint
		self.tableau = np.vstack((self.tableau, gomory_cut_constraint))
		# add cutting plane and column corresponding to the ith row of the tableau
		return
	
	def dualSimplex(self):
		self.tableau = self.tableau.astype(np.float64)
		it = 1000
		while (it):
			# check feasible
			opt, j = self.isDualOptimal()
			if (opt):
				return 0
			x_j = self.tableau[j, 1]
			l = -1
			ratio = 1e8
			for i in range(2, self.tableau.shape[1]):
				if (self.tableau[j, i] < -1e-6):
					if (ratio > -self.tableau[0, i] / self.tableau[j, i]):
						ratio = -self.tableau[0, i] / self.tableau[j, i]
						l = i
			if (l < 0):
				self.Dualstatus = "unbounded"
				self.status = "infeasible"
				return 1
			pivot_ele = self.tableau[j, l]
			# update basis index
			self.tableau[j, 0] = l - 2
			# update jth row
			self.tableau[j,1:] /= pivot_ele
			# update rows except jth
			for i in range(self.tableau.shape[0]):
				if i == j:
					continue
				self.tableau[i, 1:] -= self.tableau[j, 1:] * self.tableau[i, l]
			it -= 1

	def solve(self):
		self.solveRelaxed()
		if self.status == "infeasible":
			# print something
			print("initial_solution: Does Not Exist")
			print("final_solution: Does Not Exist")
			print("solution_status: infeasible")
			print("number_og_cuts: 0")
			print("optimal_value: Does Not Exist")
			return
		if self.status == "unbounded":
			print("initial_solution: Does Not Exist")
			print("final_solution: Does Not Exist")
			print("solution_status: unbounded")
			print("number_og_cuts: 0")
			if(self.obj):
				print("optimal_value:", -np.inf)
			else:
				print("optimal_value:", np.inf)
			return
		# when will the solution to ilp be unbounded?
		#print init_sol
		for i in range(1, self.tableau.shape[0]):
			if(self.tableau[i,0] < self.original_num_vars):
				self.bfs[int(self.tableau[i,0])] = self.tableau[i,1]
		print("initial_solution: ", end = '')
		for i in self.bfs[:-1]:
			print(i, end = ', ')
		print(self.bfs[-1])
		self.bfs = np.zeros(self.original_num_vars)
		#run algo

		integral, i = self.isInteger()
		it = 1000
		while (it > 0 and (not integral) and self.status == "optimal"):
			self.cuts += 1
			self.addCuttingPlane(i)
			self.dualSimplex()
			integral, i = self.isInteger()
			it -= 1
		if(it == 0):
			self.status = "infeasible"

		if self.status == "infeasible":
			# print something
			print("final_solution: Does Not Exist")
			print("solution_status: infeasible")
			print("number_og_cuts: 0")
			print("optimal_value: Does Not Exist")
			return
		# print fin_sol
		for i in range(1, self.tableau.shape[0]):
			if(self.tableau[i,0] < self.original_num_vars):
				self.bfs[int(self.tableau[i,0])] = self.tableau[i,1]
		print("final_solution: ", end = '')
		for i in self.bfs[:-1]:
			print(round(i), end = ', ')
		print(round(self.bfs[-1]))

		print("solution_status:", self.status)

		print("number_og_cuts:", self.cuts)

		opt_val = self.tableau[0,1]
		if(self.obj == 1):
			opt_val *= -1
		opt_val = round(opt_val)
		print("optimal_value:", opt_val)


def gomory_cut_algo():
	file = open("input_ilp.txt",'r')
	q = file.readlines()
	problem = SolveILP(0)
	if(q[1] == "minimize\n"):
		problem.obj = 1
	A = []
	i = 4
	while(q[i] != '\n'):
		A.append(eval(q[i]))
		i+=1
	problem.A = np.array(A)
	b = []
	i+=2
	while(q[i] != '\n'):
		b.append(eval(q[i]))
		i+=1
	problem.b = b

	constr = []
	i+=2
	while(q[i] != '\n'):
		if(q[i][0] == '='):
			constr.append(2)
		elif(q[i][0]=='<'):
			constr.append(1)
		else:
			constr.append(3)
		i+=1
	problem.constr = constr

	i+=2
	c = list(eval(q[i]))
	problem.c = c
	file.close()
	problem.solve()
	return



gomory_cut_algo()
