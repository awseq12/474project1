import numpy
class LogitRegression:
    def __init__(self,learning_rate: int, iterations: int):
        self.learning_rate = learning_rate
        self.iterations = iterations
        return
    def fit(self, x_train, y_train):
        #initialize weights
        weights = numpy.random.uniform(0, 1)

        c = 0
        for i in range(self.iterations):
            #call gradient descent function
            weights = self.gradient_descent(weights,  x_train, y_train)
            #calculate cost
            print(self.cost(weights, x_test, y_test))


        return
    def sigmoid(self):
        return
    def gradient_descent(self,weights, x_train, y_train ):
        result = 0
        for index, row in x_train.iterrows():
            weights
        return -result * self.learning_rate
    def cost(self, weight, x_train, y_train):
        y=[]
        sum = 0
        for index, row in x_train.iterrows():
            calc = weight(row)
            y.append(calc)
        for i in range(0,len(y_train)):
            sum += y_train[i] - y[i]

        cost = sum/len(y_train)
        return cost


    def predict(self,x_test,y_test):
        return




