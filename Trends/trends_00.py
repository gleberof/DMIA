#!/usr/bin/env python
# -*- coding: utf8 -*-


####################################################
### You are not allowed to import anything else. ###
####################################################

import numpy as np


def power_sum(l, r, p=1.0):
    """
        input: l, r - integers, p - float
        returns sum of p powers of integers from [l, r]
    """
    return np.sum(np.power(range(l,r), p))


def solve_equation(a, b, c):
    """
        input: a, b, c - integers
        returns float solutions x of the following equation: a x ** 2 + b x + c == 0
            In case of two diffrent solution returns tuple / list (x1, x2)
            In case of one solution returns one float
            In case of no float solutions return None 
            In case of infinity number of solutions returns 'inf'
    """
    if c == 0 and  b == 0 and  a == 0:
        return float('inf')

    d= b**2 - 4*a*c

    if d < 0:
        return None

    if d == 0:
        return 1.*(-b+d**.5)/(2*a)
    else:
        return (1.*(-b+d**.5)/(2*a), 1.*(-b-d**.5)/(2*a))



def replace_outliers(x, std_mul=3.0):
    """
        input: x - numpy vector, std_mul - positive float
        returns copy of x with all outliers (elements, which are beyond std_mul * (standart deviation) from mean)
        replaced with mean  
    """
    mean = np.mean(x)
    dev = np.std(x)*std_mul
    l_bound, u_bound = mean-dev, mean+dev
    res = np.array(x)
    res[res > u_bound] = mean
    res[res < l_bound] = mean
    return res


def get_eigenvector(A, alpha):
    """
        input: A - square numpy matrix, alpha - float
        returns numpy vector - any eigenvector of A corresponding to eigenvalue alpha, 
                or None if alpha is not an eigenvalue.
    """
    ws, vs = np.linalg.eig(A)
    if alpha in ws:
        i, = np.where(ws ==alpha)[0]
        return vs[i]
    else:
        return None


def discrete_sampler(p):
    """
        input: p - numpy vector of probability (non-negative, sums to 1)
        returns integer from 0 to len(p) - 1, each integer i is returned with probability p[i] 
    """
    select = np.random.random()
    p_new = np.array([ np.sum(p[:i+1]) for i in range(len(p))])
    i, =  np.where(p_new > select)
    return i[0]


def gaussian_log_likelihood(x, mu=0.0, sigma=1.0):
    """
        input: x - numpy vector, mu - float, sigma - positive float
        returns log p(x| mu, sigma) - log-likelihood of x dataset 
        in univariate gaussian model with mean mu and standart deviation sigma
    """

    return -.5*len(x)*np.log(2*np.pi*sigma**2)-.5*np.sum((x-mu)**2)/(sigma**2)


def gradient_approx(f, x0, eps=1e-8):
    """
        input: f - callable, function of vector x. x0 - numpy vector, eps - float, represents step for x_i
        returns numpy vector - gradient of f in x0 calculated with finite difference method 
        (for reference use https://en.wikipedia.org/wiki/Numerical_differentiation, search for "first-order divided difference")
    """
    ln = x.shape[0]
    xh = np.zeros((ln,ln))
    np.fill_diagonal(xh, eps)
    xh = x + xh
    return (np.apply_along_axis(f, axis = 1, arr = xh) - np.apply_along_axis(f, axis = 1, arr = np.tile(x, (ln,1))))/eps


def gradient_method(f, x0, n_steps=1000, learning_rate=1e-2, eps=1e-8):
    """
        input: f - function of x. x0 - numpy vector, n_steps - integer, learning rate, eps - float.
        returns tuple (f^*, x^*), where x^* is local minimum point, found after n_steps of gradient descent, 
                                        f^* - resulting function value.
        Impletent gradient descent method, given in the lecture. 
        For gradient use finite difference approximation with eps step.
    """
    for i in range(steps):
        x0 -= learning_rate* gradient_approx(f, x0, eps=eps)
    return (f(x0), x0)


def linear_regression_predict(w, b, X):
    """
        input: w - numpy vector of M weights, b - bias, X - numpy matrix N x M (object-feature matrix), 
        N - number of objects, M - number of features.
        returns numpy vector of predictions of linear regression model for X
        https://xkcd.com/1725/
    """
    return np.sum(X*w, axis=1) + np.tile(b, X.shape[0])


def mean_squared_error(y_true, y_pred):
    """
        input: two numpy vectors of object targets and model predictions.
        return mse
    """
    return np.mean((y_true-y_pred)**2)


def linear_regression_mse_gradient(w, b, X, y_true):
    """
        input: w, b - weights and bias of a linear regression model,
                X - object-feature matrix, y_true - targets.
        returns gradient of linear regression model mean squared error w.r.t w and b
    """
    ln = X.shape[0]
    wt = np.array(np.append(w, b), dtype=np.float64)
    X0 = np.array(np.hstack((X, np.ones((ln, 1)))), dtype=np.float64)
    res = 2. * X0.T.dot(X0.dot(wt)-y_true) / ln
    return res[:-1], res[-1]


class LinearRegressor:
    def fit(self, X_train, y_train, n_steps=1000, learning_rate=1e-2, eps=1e-8):
        """
            input: object-feature matrix and targets.
            optimises mse w.r.t model parameters 
        """
        self.w = np.zeros(X_train.shape[1], dtype=np.float64)
        self.b = 0.0
        mse = mean_squared_error(y_train, linear_regression_predict(self.w, self.b, X_train))
        self.mse = np.array(mse, dtype=np.float64)
        for i in range(n_steps):
            g_w, g_b = linear_regression_mse_gradient(self.w, self.b, X_train, y_train)
            print(i, g_w,"\n" ,g_b)
            self.w -= g_w * learning_rate
            self.b -= g_b * learning_rate
            print(self.w, "\n", self.b)
            
            new_mse = mean_squared_error(y_train, linear_regression_predict(self.w, self.b, X_train))
            
            self.mse = np.append(self.mse, new_mse)
            if abs(new_mse - mse) < eps:
                mse=new_mse
                break
            else:
                mse = new_mse
        return self


    def predict(self, X):
        return linear_regression_predict(self.w, self.b, X)
    
    def get_mse(self):
        return self.mse	

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_der(x):
    """
        returns sigmoid derivative w.r.t. x
    """
    sig = sigmoid(x)
    return sig*(1-sig)


def relu(x):
    return np.maximum(x, 0)


def relu_der(x):
    """
        return relu (sub-)derivative w.r.t x
    """
    return 0 if x <=0 else 1

class LinearRegressorMLP:
    def __init__():
        self.w = np.random.rand(X_train.shape[1]) - 0.5
        self.b = mp.random.random()  - 0.5
    
    
    def fit(self, X_train, y_train, n_steps=1000, learning_rate=1e-2, eps=1e-8):
        """
            input: object-feature matrix and targets.
            optimises mse w.r.t model parameters 
        """

        mse = mean_squared_error(y_train, linear_regression_predict(self.w, self.b, X_train))
        for i in range(n_steps):
            g_w, g_b = linear_regression_mse_gradient(self.w, self.b, X_train, y_train)
            self.w -= g_w * learning_rate
            self.b -= g_b * learning_rate
            
            new_mse = mean_squared_error(y_train, linear_regression_predict(self.w, self.b, X_train))
            
            if abs(new_mse - mse) < eps:
                mse=new_mse
                break
            else:
                mse = new_mse
        return self


    def predict(self, X):
        return linear_regression_predict(self.w, self.b, X)


class MLPRegressor:
    """
        simple dense neural network class for regression with mse loss. 
    """
    def __init__(self, n_units=[32, 32], nonlinearity=relu):
        """
            input: n_units - number of neurons for each hidden layer in neural network,
                   nonlinearity - activation function applied between hidden layers.
        """
        self.n_units = n_units
        self.nonlinearity = nonlinearity
        
        # set neurons
        self.h_layers = list()
        for i, k in enumerate(self.n_units):
            self.h_layers.append(list())
            for n in range(k):
                mlp_lr = LinearRegressorMLP()
                self.h_layers[i].append(mlp_lr)
        
        # set output layer
        self.output = LinearRegressorMLP()
        self.layer_input = np.array()
        self.input_for_output = np.array()


    def fit(self, X_train, y_train, n_steps=1000, learning_rate=1e-2, eps=1e-8):
        """
            input: object-feature matrix and targets.
            optimises mse w.r.t model parameters
            (you may use approximate gradient estimation)
        """
        
        mse = mean_squared_error(y_train, self.predict(X_train)) 
        for i in range(n_steps):
            #forward
            y_pred = self.predict(X_train)
            #backwardpropagation
            
            # output calculation
            sig_outp = (y_train - y_pred)*gradient_approx(self.nonlinearity, self.output_before_relu)
            # neurons calculation

            #weights update neurons
            
            #weights update output
            
            #check stop condition
            new_mse = mean_squared_error(y_train, self.predict(X_train)) 
            if abs(mse-new_mse) < eps:
                break
            else
                mse= new_mse

        return self


    def predict(self, X):
        """
            input: object-feature matrix
            returns MLP predictions in X
        """
        self.output_wo_relu = np.array()
        self.layer_input = np.array()
        current_signal = np.array(X)
        before_relu = np.array(X)
        for lay in self.h_layers:
            self.layer_input = np.append(self.layer_input, current_signal)
            self.output_wo_relu = np.append(self.output_wo_relu, before_relu)
            next_signal = np.array()
            before_relu = np.array()
            for n in lay:
                pred = n.predict(current_signal)
                before_relu = np.append(before_relu, pred)
                next_input = np.append(next_signal, self.nonlinearity(pred))
            current_signal = np_array(next_signal)
            
            
        self.input_for_output = np.arrya(current_signal)
        self.output_before_relu = self.output.predict(current-signal)
            
        return np.array(self.nonlinearity(self.output_before_relu))
