from __future__ import division
import numpy as np
from scipy.misc import logsumexp
import scipy.optimize

class CRF(object):

    def __init__(self,L,F):
        '''
        This class implements learning and prediction for a conditional random field.

        Args:
            L: a list of label types
            F: the number of features

        Returns:
            None
        '''

        #Your code must use the following member variables
        #for the model parameters. W_F should have dimension (|L|,F)
        #while W_T should have dimension (|L|,|L|). |L| refers to the
        #number of label types. The value W_T[i,j] refers to the
        #weight on the potential for transitioning from label L[i]
        # to label L[j]. W_F[i,j] refers to the feature potential
        #between label L[i] and feature dimension j.
        #global W_F, W_T
        global C, Lol, Fl
        self.F = F
        Lol = L
        C = len(L)
        Fl = F
        self.W_F = np.zeros((C,F))
        self.W_T = np.zeros((C,C))
        pass

    def get_params(self):
        '''
        Args:
            None

        Returns:
            (W_F,W_T) : a tuple, where W_F and W_T are the current feature
            parameter and transition parameter member variables.
        '''
        return (self.W_F, self.W_T)

    def set_params(self, W_F, W_T):
        '''
        Set the member variables corresponding to the parameters W_F and W_T

        Args:
            W_F (numpy.ndarray): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            None

        '''
        
        self.W_F = W_F
        self.W_T = W_T 
        pass

    def energy(self, Y, X, W_F=None, W_T=None):
        '''
        Compute the energy of a label sequence

        Args:
            Y (list): a list of labels from L of length T.
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            E (float): The energy of Y,X.
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        
        neg_E = 0
               
        for t in range(len(Y)):
            f = X[t]
            for c in range(C):
                if Y[t] ==  Lol[c]:
                    neg_E += np.sum(np.dot(W_F[c],f))
        
        for t in range(len(Y)-1):
            for c in range(C):
                for c_bar in range(C):
                    if Y[t] == Lol[c] and Y[t+1] == Lol[c_bar]:
                        neg_E += W_T[c,c_bar]
         
        return float(-neg_E)


    def log_Z(self, X, W_F=None, W_T=None):
        '''
        Compute the log partition function for a feature sequence X
        using the parameters W_F and W_T.
        This computation must use the log-space sum-product message
        passing algorithm and should be implemented as efficiently
        as possible.

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            log_Z (float): The log of the partition function given X

        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        T = X.shape[0]
        feature_potentials = np.zeros([T,C])
        alpha = np.zeros([T,C])
        omega = np.zeros([T,C])
        
        for t in range(T):
            f = X[t]
            for c in range(C):
                feature_potentials[t,c] = np.sum(np.dot(W_F[c],f))
        
        alpha[0] = feature_potentials[0]
        #omega[T-1] = feature_potentials[T-1]
        
        for l in range(1,T):
            for c in range(C):
                alpha[l,c] = logsumexp((alpha[l-1] + (W_T[c]) + feature_potentials[l,c]))
                #omega[T-1-l,c] = logsumexp((omega[T-l] +(W_T[c]) + feature_potentials[T-1-l,c]))
                #print('alpha',alpha[l,c],'omega', omega[T-1-l,c])
            
        Log_Z = logsumexp(alpha[T-1])
        #Log_Z_bar = logsumexp(omega[0])
        #print(Log_Z, Log_Z_bar)      
        return Log_Z

    def predict_logprob(self, X, W_F=None, W_T=None):
        '''
        Compute the log of the marginal probability over the label set at each position in a
        sequence of length T given the features in X and parameters W_F and W_T

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

       Returns:
           log_prob (numpy.ndarray): log marginal probabilties, a numpy array of type float w/ dimension (T, |L|)
           log_pairwise_marginals (numpy.ndarray): log marginal probabilties, a numpy array of type float w/ dimension (T - 1, |L|, |L|)
               - log_pairwise_marginals[t][l][l_prime] should represent the log probability of the symbol, l, and the next symbol, l_prime,
                 at time t.
               - Note: log_pairwise_marginals is a 3 dimensional array.
               - Note: the first dimension of log_pairwise_marginals is T-1 because
                       there are T-1 total pairwise marginals in a sequence in length T
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        #print(X)
        T = X.shape[0]
        log_prob = np.zeros([T,C])
        log_pairwise_marginals = np.zeros([T-1,C,C])
        feature_potentials = np.zeros([T,C])
        alpha = np.zeros([T,C])
        omega = np.zeros([T,C])

        
        for t in range(T):
            f = X[t]
            for c in range(C):
                feature_potentials[t,c] = float(np.sum(np.dot(W_F[c],f)))
        
        alpha[0] = feature_potentials[0]
        omega[T-1] = feature_potentials[T-1]
        for l in range(1,T):
            for c in range(C):
                alpha[l,c] = logsumexp((alpha[l-1] + (W_T[c]) + feature_potentials[l,c]))
                omega[T-1-l,c] = logsumexp((omega[T-l] +(W_T[c]) + feature_potentials[T-1-l,c]))

                #print('alpha',alpha[l,c],'omega', omega[T-1-l,c])
            
        Log_Z = logsumexp(alpha[T-1])
        #Log_Z_bar = logsumexp(omega[0])
        #print(Log_Z, Log_Z_bar)
        
        for l in range(T):
            for c in range(C):
                log_prob[l,c] = logsumexp(( alpha[l,c] + omega[l,c] - feature_potentials[l,c] - Log_Z))
                #print(log_prob[l,c],alpha[l,c],omega[l,c],Log_Z)
        for l in range(T-1):
            for c in range(C):
                for c_bar in range(C):
                    log_pairwise_marginals[l,c,c_bar] = logsumexp((alpha[l,c] + W_T[c,c_bar] + omega[l+1,c_bar] - Log_Z)) 
        #print('log_probs', (log_prob[0,0]))
        return (log_prob, log_pairwise_marginals)

    def predict(self, X, W_F=None, W_T=None):
        '''
        Return a list of length T containing the sequence of labels with maximum
        marginal probability for each position in an input fearture sequence of length T.

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            Yhat (list): a list of length T containing the max marginal labels given X
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        # your implementation here
        T= X.shape[0]
        Yhat = []
        
        log_probs,_ = self.predict_logprob(X,W_F,W_T) 
        for t in range(T):
            index = np.argmax(log_probs[t])
            Yhat.append(Lol[index])
        
        assert len(Yhat) == X.shape[0]

        return Yhat

    def log_likelihood(self, Y, X, W_F=None, W_T=None):
        '''
        Calculate the average log likelihood of N labeled sequences under
        parameters W_F and W_T. This must be computed as efficiently as possible.

        Args:
            Y (list): a list of length N where each element n is a list of T_n labels from L
            X (list): a list of length N where each element n is a feature array of shape (T_n,F)
            W_F (numpy.ndarray, optional): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            mll (float): the mean log likelihood of Y and X
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        mll = 0.0

        # your implementation here

        N = len(Y)

        avg = 0.0
        for i in range(N):
            Y_i = Y[i]
            X_i = X[i]
            avg += -self.energy(Y_i, X_i, W_F, W_T) -self.log_Z(X_i, W_F, W_T)
        
        mll = avg/N
        #print(mll)
        return mll

    def gradient_log_likelihood(self, Y, X, W_F=None, W_T=None):
        '''
        Compute the gradient of the average log likelihood
        given the parameters W_F, W_T. Your implementation
        must be as efficient as possible.

        Args:
            Y (list): a list of length N where each element n is a list of T_n labels from L
            X (list): a list of length N where each element n is a feature array of shape (T_n,F)
            W_F (numpy.ndarray, optional): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            (gW_F, gW_T) (tuple): a tuple of numpy arrays the same size as W_F and W_T containing the gradients

        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        gW_F = np.zeros(np.shape(self.W_F))
        gW_T = np.zeros(np.shape(self.W_T))
        N = len(Y)
        #print('Y',Y)
        #print(X)

        # your implementation here
        for i in range(N):
            X_i = X[i]
            Y_i = Y[i]
            #print('X_i',X_i,'Y_i',Y_i)
            (log_prob, log_pairwise_marginals) = self.predict_logprob(X_i,W_F,W_T)
            
            for t in range(len(Y_i)):
                F = X_i[t]
                for c in range(C):
                    if Y_i[t] ==  Lol[c]:
                        gW_F[c] += F
                    gW_F[c] -= np.multiply(F,np.exp(log_prob[t,c]))
            for t in range(len(Y_i)-1):
                for c in range(C):
                    for c_bar in range(C):
                        if Y_i[t] == Lol[c] and Y_i[t+1] == Lol[c_bar]:
                            gW_T[c,c_bar] += 1
                        gW_T[c,c_bar] -= np.exp(log_pairwise_marginals[t,c,c_bar])
                            
        assert gW_T.shape == W_T.shape
        assert gW_F.shape == W_F.shape
        #print('this', gW_F/N)
        return (-gW_F/N, -gW_T/N)


    def ll_wrapper(self, reshaped_parameters, Y ,X):
        W_F = np.reshape(reshaped_parameters[:np.product(np.shape(self.W_F))],(np.shape(self.W_F)))
        W_T = np.reshape(reshaped_parameters[np.product(np.shape(self.W_F)):],(np.shape(self.W_T)))
        mll = self.log_likelihood(Y=Y,X=X, W_F=W_F, W_T=W_T )
        return -mll
    
    def gll_wrapper(self, reshaped_parameters, Y ,X):
        W_F = np.reshape(reshaped_parameters[:np.product(np.shape(self.W_F))],(np.shape(self.W_F)))
        W_T = np.reshape(reshaped_parameters[np.product(np.shape(self.W_F)):],(np.shape(self.W_T)))
        (gW_F,gW_T) = self.gradient_log_likelihood(Y=Y,X=X, W_F=W_F, W_T=W_T )
        return np.concatenate([gW_F.reshape((C*Fl)),gW_T.reshape((C*C))])


    def fit(self, Y, X):
        '''
        Learns the CRF model parameters W_F, W_F given N labeled sequences as input.
        Sets the member variables W_T and W_F to the learned values

        Args:
            Y (list): a list of length N where each element n is a list of T_n labels from L
            X (list): a list of length N where each element n is a feature array of shape (T_n,F)

        Returns:
            None

        '''
        
        W_F, W_T = self.get_params()
        reshaped_parameters = np.zeros([(np.product(np.shape(W_F))+np.product(np.shape(W_T)))])
        result = reshaped_parameters
        #reshaped_parameters = np.concatenate(np.reshape(W_F,(1,np.product(np.shape(W_F)))),np.reshape(W_T,(1,np.product(np.shape(W_T)))))
        #print('grad_score',scipy.optimize.check_grad(self.ll_wrapper,self.gll_wrapper,reshaped_parameters,Y,X))
        #result = scipy.optimize.fmin_l_bfgs_b(func=self.ll_wrapper,x0=reshaped_parameters,fprime=self.gll_wrapper, args=(Y,X), approx_grad=0)
        result = scipy.optimize.fmin_bfgs(f=self.ll_wrapper,x0=reshaped_parameters,fprime=self.gll_wrapper, args=(Y,X))
        print('result', result)
        params = result
        self.W_F = params[:((C*Fl))].reshape([C,Fl])
        self.W_T = params[(C*Fl):].reshape([C,C])
        pass

'''
CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
         'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
         'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
         'y', 'z']
crf = CRF(CHARS,321)

print(crf.energy())
'''