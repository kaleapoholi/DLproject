import numpy as np 
import h5py

train_path='data/train_catvnoncat.h5'
test_path='data/test_catvnoncat.h5'

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def tanh(x):
    s=(np.exp(x)-np.exp(-x))/(np.exp(x)+np(-x))
    return s

def relu(x):
    s=np.maximum(0,x)
    return s

def sigmoid_grad(x):
    s= sigmoid(x)
    s_grad=s*(1-s)
    return s_grad

def tanh_grad(x):
    s=tanh(x)
    return 1-(s*s)

def relu_grad(x):
    x[x>0]=1
    x[x<=0]
    return x

def load_data(train_path,test_path):
    train_dataset=h5py.File(train_path,'r')
    train_set_X=np.array(train_dataset['train_set_x'][:])
    train_set_Y=np.array(train_dataset['train_set_y'][:])
    test_dataset=h5py.File(test_path,'r')
    test_set_X = np.array(test_dataset['test_set_x'][:])
    test_set_Y = np.array(test_dataset['test_set_y'][:])
    classes = np.array(test_dataset['list_classes'][:])

    return train_set_X, train_set_Y, test_set_X, test_set_Y,classes

def reshape_data(x_dataset, y_dataset):
    x_dataset_reshape = x_dataset.reshape((x_dataset.shape[1] * x_dataset.shape[2] * x_dataset.shape[3],x_dataset.shape[0]))
    y_dataset_reshape = y_dataset.reshape((1,y_dataset.shape[0]))
    return x_dataset_reshape, y_dataset_reshape

def initialize_parameters_with_zeros(dims):
    w=np.zeros((dims,1))
    b=0
    return w,b

def propagate(X,Y,w,b):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1- A))/m
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A - Y)/m
    cost = np.squeeze(cost)
    dicts = {'dw':dw, 'db':db}
    return dicts, cost

def optimize(X,Y, w, b, num_iterations, learning_rate):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(X,Y,w,b)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        params = {'w':w, 'b':b}
        grads = {'dw':dw, 'db':db}
    return params, grads, costs    

def predict(X_test,w,b):
    m = X_test.shape[1]
    y_prediction = np.zeros((1,m))
    w = w.reshape(X_test.shape[0],1)
    A = sigmoid(np.dot(w.T, X_test) + b)
    for i in range(A.shape[1]):
        y_prediction[0,i] = 0 if A[0,i] <= 0.5 else 1
    return y_prediction

def compute_network(X_train, Y_train, X_test, Y_test, num_iterations= 2000, learning_rate = 0.5):
    w,b = initialize_parameters_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(X_train, Y_train, w, b, num_iterations, learning_rate)    
    w = params['w']
    b = params['b']
    y_prediction_train = predict(X_train,w,b)
    y_prediction_test = predict(X_test, w, b)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,"Y_prediction_test": y_prediction_test,"Y_prediction_train" : y_prediction_train,"w" : w,"b" : b,"learning_rate" : learning_rate,"num_iterations": num_iterations}
    return d

train_set_X,train_set_Y, test_set_X, test_set_Y, classes = load_data(train_path, test_path)
print(train_set_X.shape)

# Run the function to check the errors
train_set_X_reshape, train_set_Y_reshape = reshape_data(train_set_X,train_set_Y)
test_set_X_reshape, test_set_Y_reshape = reshape_data(test_set_X,test_set_Y)
print('Train x dataset: ' + (str(train_set_X_reshape.shape)))
print('Train y dataset: ' + (str(train_set_Y_reshape.shape)))
print('Test x dataset: ' + (str(test_set_X_reshape.shape)))
print('Test y dataset: ' + (str(test_set_Y_reshape.shape)))


learning_rates = np.array([0.1, 0.01, 0.001])
for lrate in learning_rates:
    compute_network(train_set_X_reshape, train_set_Y_reshape, test_set_X_reshape, test_set_Y_reshape, num_iterations = 2000, learning_rate = lrate)