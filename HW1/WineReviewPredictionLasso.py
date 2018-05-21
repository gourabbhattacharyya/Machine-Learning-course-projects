from operator import itemgetter

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import csv

class WineReview:

    def lasso_solver(self, X,wi,w0i,y,lmbda):
        isConverged = False
        epsilon = 0.2
        currentIter = 0
        maxIter = 10
        w = wi
        w0 = w0i
        while isConverged == False and currentIter < maxIter:
            w_old = w
            w0_old = w0
            w_dum = w
            aj_val = 2*((X.T**2).sum())
            cj_val = np.matmul(X.T,y - w0 - np.matmul(X,w_dum))

            if np.any(cj_val > lmbda):
                wj_hat =(cj_val-lmbda)/aj_val
            elif np.any(cj_val < lmbda * -1):
                wj_hat = (cj_val+lmbda)/aj_val
            else:
                wj_hat = np.zeros((cj_val.shape))
            w = wj_hat

            #print "w", cj_val.shape, w.shape
            w0 = (1.0 / len(y)) * (sum(y * 1.0 - np.matmul(np.matrix(X), np.matrix(w)).getA()))
            isConverged = True
            if np.any(w_old != 0):
                if np.any(np.absolute(w - w_old)/w_old > epsilon):
                    isConverged = False

            if isConverged == True:
                if np.any(w0_old != 0):
                    if np.any(np.absolute(w0_old - w0) / w0_old > epsilon):
                        isConverged = False
            print "currentIteration", currentIter
            currentIter += 1

        return w0,w

    def RMSE(self, y,y_actual):
        return np.sqrt(np.sum(np.square(y-y_actual)) / len(y))

if __name__ == "__main__":
    # Load a text file of strings:
    featureNames = open("/Users/gourabbhattacharyya/Desktop/CSE 512/HW1/Code/kaggledata/featureTypes.txt").read().splitlines()
    # Load a csv of floats:
    x_train = io.mmread("/Users/gourabbhattacharyya/Desktop/CSE 512/HW1/Code/kaggledata/trainData.mtx").tocsc().toarray()
    x_validation = io.mmread("/Users/gourabbhattacharyya/Desktop/CSE 512/HW1/Code/kaggledata/valData.mtx").tocsc().toarray()
    x_test = io.mmread("/Users/gourabbhattacharyya/Desktop/CSE 512/HW1/Code/kaggledata/testData.mtx").tocsc().toarray()
    y_train = np.genfromtxt("/Users/gourabbhattacharyya/Desktop/CSE 512/HW1/Code/kaggledata/trainLabels.txt", delimiter=" ")
    y_validation = np.genfromtxt("/Users/gourabbhattacharyya/Desktop/CSE 512/HW1/Code/kaggledata/valLabels.txt", delimiter=" ")
    factor = 2

    x_train = np.split(x_train, 1)
    y_train = np.split(y_train, 1)
    x_validation = np.split(x_validation, 1)
    y_validation = np.split(y_validation, 1)
    x_test = np.split(x_test, 1) #[5000, 10000, 15000, 20000, len(x_test)]
    x_test_final = []
    for i in range(len(x_test)):
        if(x_test[i].shape[0] > 0):
            x_test_final.append(x_test[i])

    validationError = []
    trainError = []
    lambda_array = []
    nonNumZeroes = []
    w0List = []
    W_HAT_begin  = True
    obj = WineReview()

    for i in range(len(x_train)):
        lambda_max = 2 * max(np.absolute(np.matmul(np.transpose(x_train[i]), y_train[i] - np.average(y_train[i]))))
        print("lambda_max", lambda_max)

        iterCount = 0
        condition = True
        while condition == True and iterCount < 20:
            print 'iteration ', iterCount
            lambda_array.append(lambda_max)

            if iterCount == 0:
                w_init = np.matrix(np.zeros(x_train[i].shape[1])).T
                w_hat1 = obj.lasso_solver(x_train[i], w_init, 0, y_train[i], lambda_max)
                if W_HAT_begin:
                    W_HAT = w_hat1[1].T
                else:
                    W_HAT = np.vstack((W_HAT, w_hat1[1].T))
                w0List.append(w_hat1[0])
                nonNumZeroes.append(sum(W_HAT != 0).max())
            elif iterCount > 0:
                w_hat1 = obj.lasso_solver(x_train[i],w_hat1[1],w_hat1[0],y_train[i],lambda_max)
                W_HAT = np.vstack((W_HAT, w_hat1[1].T))
                w0List.append(w_hat1[0])
                nonNumZeroes.append(sum(w_hat1[1] != 0).max())
            validationError.append(obj.RMSE(np.matmul(x_validation[i], w_hat1[1])+w_hat1[0], y_validation[i]))
            trainError.append(obj.RMSE(np.matmul(x_train[i], w_hat1[1])+w_hat1[0], y_train[i]))

            if iterCount > 0:
                if validationError[iterCount] > validationError[iterCount - 1]:
                    if iterCount > 5:
                        condition = False
            lambda_max = lambda_max/factor
            iterCount += 1
        W_HAT_begin = False

    print "W_HAT shape : ", W_HAT.shape
    plt.plot(lambda_array, validationError, 'r.--')
    plt.plot(lambda_array, trainError, 'g.--')
    plt.xlabel('Lambda')
    plt.legend(('vldtnError','trainError'), loc='best')
    #plt.show()
    plt.savefig("lambdaVSerror.png")
    plt.clf()

    plt.plot(lambda_array, nonNumZeroes, 'b.--')
    plt.xlabel('Lambda')
    plt.ylabel('nonNumZeroes')
    #plt.show()
    plt.savefig("lambdaVSnonNumZeros.png")
    #print vldtnError, trainError, lambda_array, numNonZeroes
    bestW = 9999 if iterCount == 10000 else iterCount - 2
    print 'best lambda value achieved', lambda_array[bestW]

    file = open("BestLambda.txt", "w")
    file.write('%s' % lambda_array[bestW])
    file.close()
    print("BestLambda.txt Output file generated")


    labelPrediction = []
    count = 1;
    labelPrediction.append(("ID", "Points",))
    for i in range(len(x_test_final)):
        prediction = (np.matmul(x_test_final[i], W_HAT[bestW][0,:].T) + w0List[bestW]).tolist()

        print 'prediction data length', len(prediction)#, prediction[0]  #----------prediction

        for item in prediction:
            labelPrediction.append((count, int(min(item))),)
            count += 1;

    with open("predTestLabels.csv", "wb") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(labelPrediction)
    print("predTestLabels.csv Output file generated")


    validFeatures = []
    wFeatures = []
    weights = []
    for i in range(W_HAT.shape[1]):
        if W_HAT[bestW][0, i] != 0:
            validFeatures.append(featureNames[i])
            wFeatures.append(W_HAT[bestW][0, i])
            weights.append(np.abs(W_HAT[bestW][0, i]))

    print "Top 10 features with lowest weights", sorted(zip(validFeatures, weights, wFeatures), key=itemgetter(1))[:10]
    print "Top 10 features with largest weights", sorted(zip(validFeatures, weights, wFeatures), key=itemgetter(1), reverse=True)[:10]