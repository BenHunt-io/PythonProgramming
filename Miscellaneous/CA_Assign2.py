

# hold data
class Data:

	def __init__(self,index,data):
		self.index = index
		self.data = data



O3 = []
default = []
differences = []

i = 1

with open("ca_data.txt") as fin:
	for line in fin.readlines():
		print(line.split('    '))
		O3.append(Data(line.split('    ')[0],int(line.split('    ')[1])))
		default.append(Data(line.split('    ')[0],int((line.split('    ')[2])[:-1])))
		i+=2


for i in range(len(O3)):

	if(O3[i].data == 0 or default[i].data == 0):
		differences.append(Data(O3[i].index,O3[i].data + default[i].data))
		continue

	if(O3[i].data > default[i].data):
		differences.append(Data(O3[i].index,O3[i].data / default[i].data))
	else:
		differences.append(Data(O3[i].index,default[i].data / O3[i].data))

# print(O3)
# print(" ")
# print(default)

for i in range(len(differences)):
	print(str(differences[i].data) + " " + str(differences[i].index))



print('\n\n\n')

swap = True
while(swap):
	swap = False
	for i in range(len(differences)-1):
		if(differences[i].data < differences[i+1].data):
			temp = differences[i+1]
			differences[i+1]  = differences[i]
			differences[i] = temp
			swap = True

for i in range(len(differences)):
	print(str(differences[i].data) + " " + str(differences[i].index))