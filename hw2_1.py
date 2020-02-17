import numpy as np
import matplotlib.pyplot as plt 
import time

DISCRETE = 0
CONTINUOUS = 1

class CMAC_Map():
	def __init__(self,input_x,num_weights,num_cells):
		self.input_vector = np.linspace(np.min(input_x),np.max(input_x),num_weights-num_cells+1)
		self.num_cells = num_cells

		self.table = np.zeros((len(self.input_vector),num_weights))

		self._create()

		self.weights = np.ones((num_weights,1))

	def _create(self):
		for i in range(len(self.input_vector)):
			self.table[i,i:self.num_cells+i] = 1


class CMAC_Algorithm():
	def __init__(self,function,num_weights,num_cells,func_type):
		self.mapping = CMAC_Map(function,num_weights,num_cells)
		self.func_type = func_type

	def _error_calc(self,input_address,X,Y,e,lr,update_weights):
		upper = 0
		lower = 0

		calc_Y = np.zeros((len(Y),1)) #storage for graphing calculated function values

		for i in range(len(input_address)):
			weight_indices_0 = np.nonzero(self.mapping.table[input_address[i,0],:]) #get indices of weights corresponding to training value in look up table
			weight_indices_1 = np.nonzero(self.mapping.table[input_address[i,1],:])

			if input_address[i,0] == 0:
				output = np.sum(self.mapping.weights[weight_indices_0]) #sum weights corresponding to training value look up table address

				e = lr*(Y[i]-output)/self.mapping.num_cells #calculate error between weight sum and actual value

				if update_weights:
					self.mapping.weights[weight_indices_0] += e #update weights
			else:
				A = np.linalg.norm(self.mapping.input_vector[input_address[i,0]]-X[i])
				B = np.linalg.norm(self.mapping.input_vector[input_address[i,1]]-X[i])

				if A == 0 and B == 0:
					AB = 1
					BA = 1
				else:
					AB = A/(A+B)
					BA = B/(A+B)

				output = BA*np.sum(self.mapping.weights[weight_indices_0]) + AB*np.sum(self.mapping.weights[weight_indices_1])
				e = lr*(Y[i]-output)/self.mapping.num_cells

				if update_weights:
					self.mapping.weights[weight_indices_0] += BA*e #update weights
					self.mapping.weights[weight_indices_1] += AB*e 

			upper += abs(Y[1]-output)
			lower += Y[i]+output

			calc_Y[i] = output

		return e,upper,lower,calc_Y

	def train(self,X,Y,max_error):
		#assign look up table addresses to each training x input
		input_address = np.zeros((len(X),2))
		
		for i in range(len(X)):
			if X[i] > self.mapping.input_vector[-1]: #if training point is larger than largest value in function
				input_address[i,0] = len(self.mapping.input_vector) #set input location to max address in look up table
			elif X[i] < self.mapping.input_vector[0]: #if training point is smaller than smallest value in function
				input_address[i,0] = 0 #set input location to min address in look up table
			else:
				#create addresses for each input value
				#(largest address - 1) * (current point - min function value) / (max function value - min function value) + 1
				hold = (len(self.mapping.input_vector)-1)*(X[i]-self.mapping.input_vector[0])/(self.mapping.input_vector[-1]-self.mapping.input_vector[0])
				input_address[i,0] = np.floor(hold)

				if self.func_type == CONTINUOUS:
					input_address[i,1] = np.ceil(hold)

		#these two lines check the uniqueness of each address
		# unique,counts = np.unique(input_address,return_counts=True)
		# print(dict(zip(unique,counts)))

		input_address = input_address.astype(int)

		#Training
		lr = 0.025 #learning rate
		e = np.inf #error

		loop_count = 0
		iteration_count = 0

		start = time.time() #for measuring time to convergeance

		while e > max_error and (2*loop_count <= iteration_count):
			last_e = e 
			iteration_count += 1

			e,_,_,_ = self._error_calc(input_address,X,Y,e,lr,update_weights=True)

			_,upper,lower,_ = self._error_calc(input_address,X,Y,e,lr,update_weights=False)

			#final loop error
			# upper = 0
			# lower = 0

			# for i in range(len(input_address)):
			# 	weight_indices_0 = np.nonzero(self.mapping.table[input_address[i,0],:]) #get indices of weights corresponding to training value in look up table
			# 	weight_indices_1 = np.nonzero(self.mapping.table[input_address[i,1],:])

			# 	if input_address[i,1] == 0:
			# 		output = np.sum(self.mapping.weights[weight_indices_0])

			# 		upper += abs(Y[i]-output)
			# 		lower += Y[i]+output 
			# 	else:
			# 		A = np.linalg.norm(self.mapping.input_vector[input_address[i,0]]-X[i])
			# 		B = np.linalg.norm(self.mapping.input_vector[input_address[i,1]]-X[i])

			# 		# AB = (A/(A+B))
			# 		# BA = (B/(A+B))

			# 		if A == 0 and B == 0:
			# 			AB = 1
			# 			BA = 1
			# 		else:
			# 			AB = A/(A+B)
			# 			BA = B/(A+B)

			# 		output = BA*np.sum(self.mapping.weights[weight_indices_0]) + AB*np.sum(self.mapping.weights[weight_indices_1])

			# 		upper += abs(Y[1]-output)
			# 		lower += Y[i]+output

			e = abs(upper/lower)

			if abs(last_e - e) < 0.00001:
				loop_count += 1
			else:
				loop_count = 0

			iteration_count -= loop_count

		conv_time = time.time()-start

		#compute final error
		_,upper,lower,calc_Y = self._error_calc(input_address,X,Y,e,lr,update_weights=False)
		# upper = 0
		# lower = 0

		# calc_Y = np.zeros((len(Y),1)) #storage for graphing calculated function values

		# for i in range(len(input_address)):
		# 	weight_indices_0 = np.nonzero(self.mapping.table[input_address[i,0],:]) #get indices of weights corresponding to training value in look up table
		# 	weight_indices_1 = np.nonzero(self.mapping.table[input_address[i,1],:])

		# 	if input_address[i,1] == 0:
		# 		output = np.sum(self.mapping.weights[weight_indices_0])

		# 		upper += abs(Y[i]-output)
		# 		lower += Y[i]+output 
		# 	else:
		# 		A = np.linalg.norm(self.mapping.input_vector[input_address[i,0]]-X[i])
		# 		B = np.linalg.norm(self.mapping.input_vector[input_address[i,1]]-X[i])

		# 		# AB = (A/(A+B))
		# 		# BA = (B/(A+B))

		# 		if A == 0 and B == 0:
		# 			AB = 1
		# 			BA = 1
		# 		else:
		# 			AB = A/(A+B)
		# 			BA = B/(A+B)

		# 		output = BA*np.sum(self.mapping.weights[weight_indices_0]) + AB*np.sum(self.mapping.weights[weight_indices_1])

		# 		upper += abs(Y[1]-output)
		# 		lower += Y[i]+output

		# 	calc_Y[i] = output

		error = abs(upper/lower)

		return calc_Y,error,conv_time

	def test(self,X,Y):
		#assign look up table addresses to each training x input
		input_address = np.zeros((len(X),2))
		
		for i in range(len(X)):
			if X[i] > self.mapping.input_vector[-1]: #if training point is larger than largest value in function
				input_address[i,0] = len(self.mapping.input_vector) #set input location to max address in look up table
			elif X[i] < self.mapping.input_vector[0]: #if training point is smaller than smallest value in function
				input_address[i,0] = 0 #set input location to min address in look up table
			else:
				#create addresses for each input value
				#(largest address - 1) * (current point - min function value) / (max function value - min function value) + 1
				hold = (len(self.mapping.input_vector)-1)*(X[i]-self.mapping.input_vector[0])/(self.mapping.input_vector[-1]-self.mapping.input_vector[0])
				input_address[i,0] = np.floor(hold)

				if self.func_type == CONTINUOUS:
					input_address[i,1] = np.ceil(hold)

		#these two lines check the uniqueness of each address
		# unique,counts = np.unique(input_address,return_counts=True)
		# print(dict(zip(unique,counts)))

		input_address = input_address.astype(int)

		#compute accuracy
		upper = 0
		lower = 0

		calc_Y = np.zeros((len(Y),1)) #storage for graphing calculated function values

		for i in range(len(input_address)):
			weight_indices_0 = np.nonzero(self.mapping.table[input_address[i,0],:]) #get indices of weights corresponding to training value in look up table
			weight_indices_1 = np.nonzero(self.mapping.table[input_address[i,1],:])

			if input_address[i,1] == 0:
				output = np.sum(self.mapping.weights[weight_indices_0])

				upper += abs(Y[i]-output)
				lower += Y[i]+output 
			else:
				A = np.linalg.norm(self.mapping.input_vector[input_address[i,0]]-X[i])
				B = np.linalg.norm(self.mapping.input_vector[input_address[i,1]]-X[i])

				# AB = (A/(A+B))
				# BA = (B/(A+B))

				if A == 0 and B == 0:
					AB = 1
					BA = 1
				else:
					AB = A/(A+B)
					BA = B/(A+B)

				output = BA*np.sum(self.mapping.weights[weight_indices_0]) + AB*np.sum(self.mapping.weights[weight_indices_1])

				upper += abs(Y[1]-output)
				lower += Y[i]+output

			calc_Y[i] = output

		error = abs(upper/lower)
		accuracy = 100-error

		return calc_Y,accuracy,error


def output_map(table):
	with open('lut.txt','w') as filehandle:
		for line in table:
			for item in line:
				filehandle.write('%s,' % str(item))
			filehandle.write('\n')


if __name__ == "__main__":

	X = np.linspace(0,10,100)
	Y = np.sin(X)

	rand_points = np.random.randint(0,99,100)

	train_x = X[rand_points[:70]]
	train_y = Y[rand_points[:70]]

	test_x = X[rand_points[70:]]
	test_y = Y[rand_points[70:]]
	
	num_cells = np.arange(1,35,1)

	#DISCRETE
	d_train_error = []
	d_test_error = []
	d_conv_time = []
	d_accuracy = []

	print('DISCRETE')
	#iterate through different cell overlaps from 1 to 34
	for i in range(1,35):

		cmac = CMAC_Algorithm(X,35,i,DISCRETE)
		# output_map(cmac.mapping.table)

		training_Y,training_error,conv_time = cmac.train(train_x,train_y,0)
		testing_Y,testing_accuracy,testing_error = cmac.test(test_x,test_y)

		d_train_error.append(training_error)
		d_test_error.append(testing_error)
		d_conv_time.append(conv_time)
		d_accuracy.append(testing_accuracy)

		print("Number of Cells: %d | Training Error: %f | Testing Error: %f | Testing Accuracy: %f | Time to Convergeance: %f" %(i,training_error,testing_error,testing_accuracy,conv_time))

	#CONTINUOUS
	c_train_error = []
	c_test_error = []
	c_conv_time = []
	c_accuracy = []

	print('CONTINUOUS')
	#iterate through different cell overlaps from 1 to 34
	for i in range(1,35):
		
		cmac = CMAC_Algorithm(X,35,i,CONTINUOUS)
		# output_map(cmac.mapping.table)

		training_Y,training_error,conv_time = cmac.train(train_x,train_y,0)
		testing_Y,testing_accuracy,testing_error = cmac.test(test_x,test_y)

		c_train_error.append(training_error)
		c_test_error.append(testing_error)
		c_conv_time.append(conv_time)
		c_accuracy.append(testing_accuracy)

		print("Number of Cells: %d | Training Error: %f | Testing Error: %f | Testing Accuracy: %f | Time to Convergeance: %f" %(i,training_error,testing_error,testing_accuracy,conv_time))

	plt.figure('Training Error')
	plt.plot(num_cells,d_train_error)
	plt.plot(num_cells,c_train_error)
	plt.legend(["Discrete","Continuous"])

	plt.figure('Testing Error')
	plt.plot(num_cells,d_test_error)
	plt.plot(num_cells,c_test_error)
	plt.legend(["Discrete","Continuous"])

	plt.figure('Time to Convergeance')
	plt.plot(num_cells,d_conv_time)
	plt.plot(num_cells,c_conv_time)
	plt.legend(["Discrete","Continuous"])

	plt.figure('Accuracy')
	plt.plot(num_cells,d_accuracy)
	plt.plot(num_cells,c_accuracy)
	plt.legend(["Discrete","Continuous"])
	
	# #some nonsense to sort training and testing values for plotting

	# #training value sorting
	# train_pairs = {}
	# for i in range(len(train_x)):
	# 	train_pairs[train_x[i]] = training_Y[i][0]

	# sorted_x_train = []
	# sorted_y_train = []
	# for key in sorted(train_pairs.keys()):
	# 	sorted_x_train.append(key)
	# 	sorted_y_train.append(train_pairs[key])

	# #testing value sorting
	# test_pairs = {}
	# for i in range(len(test_x)):
	# 	test_pairs[test_x[i]] = testing_Y[i][0]

	# sorted_x_test = []
	# sorted_y_test = []
	# for key in sorted(test_pairs.keys()):
	# 	sorted_x_test.append(key)
	# 	sorted_y_test.append(test_pairs[key])

	# #plotting for comparison
	# plt.plot(X,Y)
	# plt.plot(sorted_x_train,sorted_y_train)
	# plt.plot(sorted_x_test,sorted_y_test)
	# # plt.scatter(train_x,training_Y,color='red')
	# # plt.scatter(test_x,testing_Y,color='green')
	plt.show()