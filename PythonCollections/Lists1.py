import numpy as np

list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
print(list)
print(list[1:])
print(list[:1])
print(list[::2]) #Striding by 2

subList1, *subList2 = list[::2]  # * <-- unpacking operator
print(subList1)
print(subList2)

