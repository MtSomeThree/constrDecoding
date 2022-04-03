class LogicalConstraintFunction(object):
	def __init__(self, idx):
		self.idx = idx

	def set_constraint_id(self, idx):
		self.idx = idx

	def constraint_function1(self, sentence, input=None): #6439, 6308 || 3930
		flag1 = 0
		flag2 = 0
		flag3 = 0
		for x in sentence:
			if x % 16 == 0:
				flag1 = 1
			if x % 16 == 1:
				flag2 = 1
			if x % 16 == 2:
				flag3 = 1
		if flag1 + flag2 + flag3 == 3:
			return True
		return False 

	def constraint_function2(self, sentence, input=None): 
		for x in sentence:
			if x % 64 == 0:
				return True
		return False

	def __call__(self, sentence, input=None):
		if self.idx == 1:
			return self.constraint_function1(sentence, input)
		if self.idx == 2:
			return self.constraint_function2(sentence, input)

def constraint_function1(sentence, input=None): #6439, 6308 || 3930
	flag1 = 0
	flag2 = 0
	flag3 = 0
	for x in sentence:
		if x % 16 == 0:
			flag1 = 1
		if x % 16 == 1:
			flag2 = 1
		if x % 16 == 2:
			flag3 = 1
	if flag1 + flag2 + flag3 == 3:
		return True
	return False

def constraint_function2(sentence, input=None): 
	for x in sentence:
		if x % 64 == 0:
			return True
	return False

def get_constraint_function(id):
	if id == 1:
		return constraint_function1
	if id == 2:
		return constraint_function2