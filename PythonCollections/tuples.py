
myList = []
cls_dict = {} #empty dict

#regular 1 tuple
for i in range(20):
	myList.append(("list", i))


cls_dict[35] = "Washing Dishes"
print(35 in cls_dict)
print(36 in cls_dict)

#Print the tuples that are in the list
for entry in myList:
	print(entry[0] + " " + str(entry[1]))


