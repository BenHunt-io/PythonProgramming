import numpy as np

# x = np.zeros((1,79))

# print(len(x[0]))
# print(x)


arr = np.full((1,79,224,224,2),1)

arr = arr[0][0]
print(arr.shape)
arr[0][0][0] = 50
arr[0][0][1] = 25
print()
print()


flowx = arr[:,:,0]
flowy = arr[:,:,1]

print(flowx)
print(flowx.shape)
print(flowy)
print(flowy.shape)
