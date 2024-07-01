import numpy as np


class optimize:
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
		self.original_num_vars = 0
		self.num_vars = 0
		self.bfs = 0
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
		i = 0
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
				if (self.tableau[i, j] > 0):
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

		# if (self.tableau[0][1] != 0): #changed here @ वत्सल 
		#     self.status = "infeasible"
        # removing redundant rows
        # need_to_remove=[]
        # variable_until=len(self.tableau)-2
        # for i in range(1,len(self.tableau)):
        #     if self.tableau[i][0]>3: #why 3 here??? @रोहित 
        #         var=variable_until+1
        #         count=0
        #         for j in range(2,var+2):
        #             if(self.tableau[i][j]==0):
        #                 count+=1
        #         if (count==var):
        #             need_to_remove.append(self.tableau[i][0])
        # new_tableau=np.array(self.tableau[0])
        
        # if(len(need_to_remove) > 0):
        #     for i in range(1,len(self.tableau)):
        #         if self.tableau[i][0] in need_to_remove:
        #             continue
        #         else:
        #             new_tableau=np.vstack([new_tableau,self.tableau[i]])
        # self.tableau=new_tableau
        # variable_until=variable_until+2
        # new_tableau=np.array([])
        # for j in range(len(self.tableau[0])):
        #     if(j<=variable_until):
        #         new_tableau=np.append(new_tableau,self.tableau[0][j])
        # for i in range(1,len(self.tableau)):
        #     new_array=np.array([])
        #     for j in range(len(self.tableau[i])):
                
        #         if(j<=variable_until):
        #             new_array=np.append(new_array,self.tableau[i][j])
                
        #     new_tableau=np.vstack([new_tableau,new_array])
        # self.tableau=new_tableau
		m, n = self.tableau.shape[0]-1,self.num_vars
		self.tableau = self.tableau[:,:n+2]
		i = 1
		while(i<self.tableau.shape[0]):
			if self.tableau[i,0]>=n:
				f = 1
				for j in range(2,n+2):
					if (self.tableau[i,j] != 0):
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
	        if (self.tableau[0, i] < 0):
	            return False, i
	    self.status = "optimal"
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
	            if (self.tableau[i, j] > 0):
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

	def solve(self):
    	# driver code has taken obj, c, A, b, constr vectors
    	# incase of infeasible and unbounded lets assume we dont care about the final bfs 
		d = {}

		self.standardize()

		self.tableau_maker_phase1()
		initial_tableau = self.tableau[1:,1:].copy()
		d["initial_tableau"] = initial_tableau

		self.createTableau()
		final_tableau = self.tableau[1:,1:].copy()
		d["final_tableau"] = final_tableau

		if(self.status == "infeasible"):
			d["solution_status"] = "infeasible"
			d["optimal_solution"] = "Does Not Exist"
			d["optimal_value"] = "Does Not Exist"
		
		else:
			self.phase2TableauSolver()
			final_tableau = self.tableau[1:,1:].copy()
			optimal_value = self.tableau[0,1]
			
			d["final_tableau"] = final_tableau
			d["solution_status"] = self.status
			
			for i in range(1,self.tableau.shape[0]):
				if(int(self.tableau[i,0]) < self.original_num_vars):
					self.bfs[int(self.tableau[i,0])] = self.tableau[i,1]
			d["optimal_solution"] = self.bfs
			d["optimal_value"] = optimal_value

			if(self.status == "unbounded"):
				if(self.obj):
					d["optimal_solution"] = "Does Not Exist"
					d["optimal_value"] = -np.inf
					
				else:
					d["optimal_solution"] = "Does Not Exist"
					d["optimal_value"] = np.inf
					
				
			else:
				if(self.obj):
					d["optimal_value"] *= -1 #in case of min we need to invert the sign 
		return d

def simplex_algo():
	file = open("input.txt",'r')
	q = file.readlines()
	problem = optimize(0)
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

	d = problem.solve()
	return d
# problem = optimize(1)
# c = [1, 1, 1,1]
# A = np.array([[1, 2, 3,0], [-1, 2, 6, 0], [0 ,4 , 9,0],[0,0,3,1]])
# b = [3, 2, 5, 1]

# constr = [2, 2, 2,2]
# problem.A = A
# problem.b = b
# problem.c = c
# problem.constr = constr

# print(problem.solve())


# print(simplex_algo())
