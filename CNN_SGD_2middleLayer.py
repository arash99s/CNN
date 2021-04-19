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
NUM_MIDDLE_LAYER1_NORONS = 10
NUM_MIDDLE_LAYER2_NORONS = 10
NUMBER_EPOCHS = 300

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
    file = open("humps.txt", "r")
    for line in file:
        x0 = float(line.split()[0])
        y0 = float(line.split()[1])
        if x0 == x:
            return y0
    return -1

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
x_valid[:,1] = total_input[30:0:-1,0]
################# test input
x_test[:,0] = total_input[0:10,0]
x_test[:,1] = total_input[10:20,0]

############## calcute d_learn
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
eta = 0.04
epsilon = 0.01
epoch = 0
epoch_errors = np.zeros((NUMBER_EPOCHS , NUMBER_OF_X_LEARN))
validation_errors = [] 
learning_errors = []
input_range = np.array(range(NUMBER_OF_X_LEARN))
e_learn_sum = 0
e_valid_sum = 10
while(epoch < NUMBER_EPOCHS and e_valid_sum > epsilon):
    net1 = np.ones((NUM_MIDDLE_LAYER1_NORONS,1))
    net1_valid = np.ones((NUM_MIDDLE_LAYER1_NORONS,1))
    o1 = np.ones((NUM_MIDDLE_LAYER1_NORONS,1))
    o1_valid = np.ones((NUM_MIDDLE_LAYER1_NORONS,1))
    diff_o1 = np.ones((NUM_MIDDLE_LAYER1_NORONS,1))
    
    for m in input_range:
        ######################################### Feedforward
        net1[:,0] = w1[:,0].dot(x_learn[m,0]) + w1[:,1].dot(x_learn[m,1]) 
        net1 += wb1
        ########### Activation
        o1 = matrix_function(net1, logsig) # f1(net1)
        diff_o1 = matrix_function(net1, diff_logsig) #diff_f1(net1)
        net2 = w2.dot(o1) + wb2  # (n2*1)
        o2 = matrix_function(net2, logsig) # f2(net2)
        diff_o2 = matrix_function(net2, diff_logsig) #diff_f2(net2)
        net3 = w3.dot(o2) + wb3
        o3 = net3[0,0]
        e = d_learn[m , 0] - o3
        epoch_errors[epoch , m] = e # store error of learning
        ######################################### Backpropagation
        w3 += eta * e * o2.T  # diff_activationFunction3 = 1
        wb3 += eta * e
        e2 = w3.T * e # (n2*1)
        delta2 = e2 * diff_o2 # (n2*1)
        w2 += eta * delta2.dot(o1.T) # (n2*n1)
        wb2 += eta * delta2 # (n2*1)
        e1 = (w2.T).dot(e2) # (n1*1)
        delta1 = e1 * diff_o1 # (n1*1)
        xin = x_learn[m]
        xin.shape = (2,1)
        w1 += eta * delta1.dot(xin.T) # (n1*2)
        wb1 += eta * delta1
    ######################### learning/validation error
    ## learning
    net1_learn = w1.dot(x_learn.T) + wb1
    o1_learn = matrix_function(net1_learn, logsig)
    net2_learn = w2.dot(o1_learn) + wb2
    o2_learn = matrix_function(net2_learn, logsig)
    net3_learn = w3.dot(o2_learn) + wb3
    o3_learn = net3_learn
    e_learn = d_learn.T - o3_learn
    e_learn_sum = calcute_E(e_learn)
    # validation
    net1_valid = w1.dot(x_valid.T) + wb1
    o1_valid = matrix_function(net1_valid, logsig)
    net2_valid = w2.dot(o1_valid) + wb2
    o2_valid = matrix_function(net2_valid, logsig)
    net3_valid = w3.dot(o2_valid) + wb3
    o3_valid = net3_valid
    e_valid = d_valid.T - o3_valid
    e_valid_sum = calcute_E(e_valid)

    validation_errors.append(e_valid_sum)
    learning_errors.append(e_learn_sum)
    
    
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