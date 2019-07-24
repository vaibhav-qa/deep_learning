import numpy as np 
train_data = np.array([[1.0,1.0]])
train_targets = np.array([2.0])
print(train_data)
for i in range(3,10000,2):
    train_data= np.append(train_data,[[i,i]],axis=0)
    train_targets= np.append(train_targets,[i+i])
print(train_data)
print(train_targets)
test_data = np.array([[2.0,2.0]])
test_targets = np.array([4.0])
for i in range(4,8000,4):
    test_data = np.append(test_data,[[i,i]],axis=0)
    test_targets = np.append(test_targets,[i+i])
print(test_data)
print(test_targets)
