import numpy as np
import matplotlib.pyplot as plt
import math 
import random
import decimal
decimal.getcontext().prec = 100

NUMBER_OF_TOTAL_INPUT = 100
NUMBER_OF_X_LEARN = 900
NUMBER_OF_X_VALID = 30
NUMBER_OF_X_TEST = 10
NUM_MIDDLE_LAYER1_NORONS = 5
NUM_MIDDLE_LAYER2_NORONS = 5
NUMBER_EPOCHS = 50
eta = 0.0001
epsilon = 0.01

def tansig(x):
    output = (1-decimal.Decimal(math.exp(-2*x)))/(1+decimal.Decimal(math.exp(-2*x)))
    return float(output)

def diff_tansig(x):
    output = (4*decimal.Decimal(math.exp(-2*x)))/((1+decimal.Decimal(math.exp(-2*x)))**2)
    return float(output)

def logsig(x):
    output = 1/(1+decimal.Decimal(math.exp(-1*x)))
    return float(output)

def diff_logsig(x):
    output = (decimal.Decimal(math.exp(-1*x)))/((1+decimal.Decimal(math.exp(-1*x)))**2)
    return float(output)

def humps(x):
    return (1 / ((x - 0.3) ** 2 + 0.01)) + (1 / ((x - 0.9) ** 2 + 0.04)) - 6

def target_function(x , y):
    return x**2 + y**2 + humps(x)

def matrix_function(mat , func):
    output = np.ones(mat.shape)
    for idx, x in np.ndenumerate(mat):
        #print(idx, x)
        output[idx] = func(x)
    return output

def calcute_E(arr):
    output = 0
    for element in arr.ravel():
        output += element * element
    return 0.5 * output

total_input = np.ones((NUMBER_OF_TOTAL_INPUT,1))
x_learn = np.ones((NUMBER_OF_X_LEARN,2))
x_valid = np.ones((NUMBER_OF_X_VALID,2))
x_test = np.ones((NUMBER_OF_X_TEST,2))

total_input[:,0] = np.arange(0,NUMBER_OF_TOTAL_INPUT*0.01,0.01)

################## random training input
x1 = np.ones((30 , 1))
x1[: , 0] = total_input[0:30,0]
x2 = np.ones((30 , 0))
for i in range(30):
    x2 = np.hstack((x2,x1))
list_x = x2.ravel()
random.shuffle(list_x)
x_learn[: , 0] = list_x.copy()
random.shuffle(list_x)
x_learn[: , 1] = list_x.copy()

################# validation input
x_valid[:,0] = total_input[30:0:-1,0]
x_valid[:,1] = total_input[0:30:1,0]
################# test input
x_test[:,0] = total_input[0:10,0]
x_test[:,1] = total_input[10:20,0]

############## calculate targets
## learn
d_learn = np.ones((NUMBER_OF_X_LEARN,1))
for index in range(NUMBER_OF_X_LEARN):
    d_learn[index , 0] = target_function(x_learn[index,0], x_learn[index,1])

d_learn_max = max(d_learn)
## validation
d_valid = np.ones((NUMBER_OF_X_VALID,1))
for index in range(NUMBER_OF_X_VALID):
    d_valid[index , 0] = target_function(x_valid[index,0], x_valid[index,1])

d_valid_max = max(d_valid)
## test
d_test = np.ones((NUMBER_OF_X_TEST,1))
for index in range(NUMBER_OF_X_TEST):
    d_test[index , 0] = target_function(x_test[index,0], x_test[index,1])

d_test_max = max(d_test)
## max for normalization
d_max = max([d_learn_max , d_valid_max, d_test_max])

d_learn = d_learn / d_max
d_valid = d_valid / d_max
d_test = d_test / d_max
############### first weights w1,w1b,w2,w2b,w3,w3b

w1 = np.ones((NUM_MIDDLE_LAYER1_NORONS,2))
wb1 = np.ones((NUM_MIDDLE_LAYER1_NORONS,1))
w2 = np.ones((NUM_MIDDLE_LAYER2_NORONS,NUM_MIDDLE_LAYER1_NORONS))
wb2 = np.ones((NUM_MIDDLE_LAYER2_NORONS,1))
w3 = np.ones((1,NUM_MIDDLE_LAYER2_NORONS))
wb3 = np.ones((1,1))

for i in range(NUM_MIDDLE_LAYER1_NORONS):
    w1[i , 0] = random.random()
    w1[i , 1] = random.random()
    wb1[i, 0] = random.random()
    for j in range(NUM_MIDDLE_LAYER2_NORONS):
        w2[j , i] = random.random()
        wb2[j , 0] = random.random()
        w3[0 , j] = random.random()
wb3[0,0] = random.random()

###############

epoch = 0
epoch_errors = np.zeros((NUMBER_EPOCHS , NUMBER_OF_X_LEARN))
validation_errors = [] 
learning_errors = []
input_range = np.array(range(NUMBER_OF_X_LEARN))
e_learn_sum = 0
e_valid_sum = 10
while(epoch < NUMBER_EPOCHS and e_valid_sum > epsilon):
    # learning Feedforward
    net1_learn = w1.dot(x_learn.T) + wb1 # (n1*x)
    o1_learn = matrix_function(net1_learn, logsig)
    diff_o1 = matrix_function(net1_learn, diff_logsig) # (n1*x)

    net2_learn = w2.dot(o1_learn) + wb2 # (n2*x)
    o2_learn = matrix_function(net2_learn, logsig)
    diff_o2 = matrix_function(net2_learn, diff_logsig) # (n2*x)

    net3_learn = w3.dot(o2_learn) + wb3 # (1*x)
    o3_learn = net3_learn
    e_learn = d_learn.T - o3_learn # (1*x)
    e_learn_sum = calcute_E(e_learn)
    # Backpropagation
    w3 += eta * e_learn.dot(o2_learn.T)  # (1*n2)
    wb3 += eta * e_learn.dot(np.ones((NUMBER_OF_X_LEARN,1))) # (1*1)
    e2 = (w3.T).dot(e_learn) # (n2*x)
    delta2 = e2 * diff_o2 # (n2*x)

    w2 += eta * delta2.dot(o1_learn.T) # (n2*n1)
    wb2 += eta * delta2.dot(np.ones((NUMBER_OF_X_LEARN,1))) # (n2*1)
    e1 = (w2.T).dot(e2) # (n1*x)
    delta1 = e1 * diff_o1 # (n1*x)

    w1 += eta * delta1.dot(x_learn) # (n1*2)
    wb1 += eta * delta1.dot(np.ones((NUMBER_OF_X_LEARN,1))) # (n1*1)
    # validation
    net1_valid = w1.dot(x_valid.T) + wb1
    o1_valid = matrix_function(net1_valid, logsig)
    net2_valid = w2.dot(o1_valid) + wb2
    o2_valid = matrix_function(net2_valid, logsig)
    net3_valid = w3.dot(o2_valid) + wb3
    o3_valid = net3_valid
    e_valid = d_valid.T - o3_valid
    e_valid_sum = calcute_E(e_valid)

    learning_errors.append(e_learn_sum/NUMBER_OF_X_LEARN)
    validation_errors.append(e_valid_sum/NUMBER_OF_X_VALID)
    
    
    #print("epoch : " , epoch)
    epoch += 1
    #random.shuffle(input_range)


#print(learning_errors)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#print(validation_errors)

####### test error
net1_test = w1.dot(x_test.T) + wb1
o1_test = matrix_function(net1_test, logsig)
net2_test = w2.dot(o1_test) + wb2
o2_test = matrix_function(net2_test, logsig)
net3_test = w3.dot(o2_test) + wb3
o3_test = net3_test
e_test = d_test.T - o3_test

print(e_test)
####### display charts 
plt.plot(range(len(validation_errors)) , validation_errors , 'r' , label="valdiation error")
plt.plot(range(len(learning_errors)) , learning_errors , 'g' , label="learning error")
plt.xlabel("epoch")
plt.legend()
plt.show()