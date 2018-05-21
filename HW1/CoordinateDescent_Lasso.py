import numpy as np
import matplotlib.pyplot as plt

class CoordinateDescent:

    def generateDataSet(self, N,d,k,sigma):
        X_val = np.random.randn(N,d)
        w_act = np.matrix(np.zeros(d)).T
        for i in range(k):
            if (np.random.randint(1,3) == 1):
                wi = 10
            else:
                wi = -10
            w_act[i] = wi

        epsilon = sigma * np.random.randn(N,1)
        y_val = np.matrix(X_val)*w_act + epsilon
        return X_val,y_val,w_act


    def calculatePrecisionRecall(self, k,d, W_HAT):
        precision = np.zeros(len(W_HAT))
        recall = np.zeros(len(W_HAT))
        for i in range(len(W_HAT)):
            correctVal = 0
            tVal = 0
            for index in range(d):
                if  W_HAT[i][0,index] != 0:
                    if	index < k:
                        correctVal += 1
                    tVal += 1

            if tVal != 0:
                precision[i] = correctVal * 1.0 / tVal
            else:
                precision[i] = 1

            if k != 0:
                recall[i] = correctVal*1.0 / k
        return precision, recall


    def lasso_solver(self, X,wi,w0i,y,lmbda):
        isConverged = False
        epsilon = 0.2
        currentIter = 0
        maxIter = 10
        n = len(X.T[0])
        d = len(wi)
        w = wi
        w0 = w0i
        while isConverged == False and currentIter < maxIter:
            w_old = w
            w0_old = w0
            for j in range(d):
                w_dum = w
                w_dum[j] = 0
                aj_val = 2*((X.T[j]**2).sum())
                jth_col = np.reshape(X[:,j],[n,1])
                cj_val = np.dot(jth_col.T,y - w0 - np.dot(X,w_dum))

                if cj_val > lmbda:
                    wj_hat =(cj_val-lmbda)/aj_val #----------changed added max
                elif cj_val < lmbda * -1: #----------changed
                    wj_hat = (cj_val+lmbda)/aj_val   #----------changed added min
                else:
                    wj_hat = 0

                w[j] = wj_hat
            w0 = (1.0/len(y))*(sum(y*1.0-(np.matrix(X)*np.matrix(w)).getA()))
            isConverged = True
            for i in range(d):
                if np.any(w_old[i] != 0):
                    if np.absolute(w[i] - w_old[i])/w_old[i] > epsilon:
                        isConverged = False
                        break

            if isConverged == True:
                if np.any(w0_old != 0):
                    if np.absolute(w0_old - w0)/w0_old > epsilon:
                        isConverged = False
            #print "currentIteration", currentIter
            currentIter += 1
        return w0,w


    def generateSyntheticDataSet1(self, N=40, d=65, k=5, sigma=1, factor=2):
        X, y, w_act = self.generateDataSet(N, d, k, sigma)
        y_bar = np.mean(y)
        lambda_max = 2* max(np.absolute(np.matrix(X).T*(y-y_bar)))
        count = 0;

        while count < 10:
            if count == 0:
                lambdaStore = np.array(lambda_max)
                w_int = np.matrix(np.zeros(d)).T
                w_hat1 = self.lasso_solver(X, w_int, 0, y, lambda_max)
                W_HAT = w_hat1[1].T

            elif count > 0:
                lambdaStore = np.hstack((lambdaStore, lambda_max))
                w_hat1 = self.lasso_solver(X, w_hat1[1], w_hat1[0], y, lambda_max)
                W_HAT = np.vstack((W_HAT, w_hat1[1].T))

            lambda_max = lambda_max / factor
            count += 1;

        return W_HAT, lambdaStore.getA()[0]


    def generateSyntheticDataSet2(self, lambdasArray, N=40, d=65, k=5, sigma=10):
        X, y, w_act = self.generateDataSet(N, d, k, sigma)
        w_int = np.matrix(np.zeros(d)).T
        count = 0

        while count < len(lambdasArray):
            w_hat1 = self.lasso_solver(X, w_int, 0, y, lambdasArray[count])

            if count == 0:
                W_HAT = w_hat1[1].T

            elif count > 0:
                W_HAT = np.vstack((W_HAT, w_hat1[1].T))

            count += 1

        return W_HAT


if __name__ == "__main__":
    NoOfRec=250
    d=80
    k=10
    sigma=1

    obj = CoordinateDescent()
#----------------Part1 for Ques 4.3 Begin----------------:
    W_HAT, lambdaStoreVal = obj.generateSyntheticDataSet1(NoOfRec,d,k,sigma,2)

    precision, recall = obj.calculatePrecisionRecall(k, d, W_HAT)
    # print precision, recall

    plt.plot(lambdaStoreVal, precision, 'r.--')
    plt.plot(lambdaStoreVal, recall, 'yo--')
    plt.xlabel('Lambda')
    plt.legend(('Precision', 'Recall'), loc='best')
    # plt.show()
    plt.savefig("lambdaVSPrecision_Recall_1.png")
    plt.clf()

#----------------Part1Ends----------------

#----------------Part2 for Ques 4.3----------------:
    for i in range(10):
        W_HatLambdas = obj.generateSyntheticDataSet2(lambdaStoreVal, NoOfRec, d, k, 10)
        precision,recall = obj.calculatePrecisionRecall(k, d, W_HatLambdas)
        #print(precision, recall)
        plt.plot(lambdaStoreVal, precision, 'r')
        plt.plot(lambdaStoreVal, recall, 'g')
    plt.xlabel('Lambda')
    plt.ylabel('Precisions -- Red lines and Recall -- green lines')
    #plt.show()
    plt.savefig("lambdaVSPrecision_Recall_2.png")
#----------------Part2Ends----------------