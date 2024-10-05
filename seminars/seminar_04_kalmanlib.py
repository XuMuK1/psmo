import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from tqdm import tqdm

class SignalGenerator:

    def __init__(self):
        pass

    def generate(self):
        raise NotImplementedError
    

class LinearGaussianSignalGenerator(SignalGenerator):

    def __init__(self, A, B, Ry, Rx,d=1,v=1):
        '''
        The model is 
        Y_{t+1} = AY_t + U_t (unobservable)
        X_{t+1} = BY_{t+1} + W_t (observable)
        '''
        self.A=A
        self.B=B
        self.Ry=Ry
        self.Rx=Rx
        self.d=d
        self.v=v

        #add shape checks here


    def generate(self, N, start):
        '''
        Generates signal and measurements
        int N -- signal length
        float[] start -- start state (Y_0)
        '''
        Ys = np.zeros([self.A.shape[0],N]) #signal
        Xs = np.zeros([self.B.shape[0],N]) #measurements

        Ys[:,0] = start
        Xs[:,0] = self.measurement(Ys[:,0])
        for i in np.arange(1,N):
            Ys[:,i] = self.evolution(Ys[:,i-1])
            Xs[:,i] = self.measurement(Ys[:,i])

        return Ys, Xs
    

    def measurement(self, y):
        '''
        Measures the signal
        '''
        return self.B @ y + self.v*np.random.multivariate_normal(np.zeros([self.B.shape[0]]), self.Rx)
    
    def evolution(self, y):
        '''
        Updates the signal wrto dynamics
        '''
        return self.A @ y + self.d*np.random.multivariate_normal(np.zeros([self.A.shape[0]]), self.Ry)
    

class KalmanFilter:

    def __init__(self,A,B,Ry,Rx,startMean,startCov):
        """Kalman Filter for Linear Dynamic System of type

        Y(t+1) = AY(t) + U(t),      U(t)  iid  N(0,Ry)    (latent)
        X(t+1) = BY(t+1) + W(t),    W(t)  iid  N(0,Rx)    (observable)
        Y(0) ~ N(startMean, startCov)
        
        Basically, anything can be estimated but beware of divergence of EM-algorithm.
        Args:
            A (ndarray (d,d) ): dynamics matrix
            B (ndarray (k,d) ): observation matrix
            Ry (ndarray (d,d)): dynamics noise covariance
            Rx (ndarray (k,k)): observation noise covariance
            startMean (ndarray (d,)): dynamics start mean
            startCov (ndarray (d,d)): dynamics start covariance
        """        
        self.A = A
        self.B = B
        self.Rx = Rx
        self.Ry = Ry
        self.startMean = startMean
        if(startCov is None):
            self.startCov = np.eye(self.A.shape[0]) 
        else:
            self.startCov = startCov
            
    def __str__(self):
        return f"KalmanFilter,\n    A=\n    {self.A},\n    B=\n    {self.B},\n    Rx=\n    {self.Rx},\n    Ry=\n    {self.Ry},\n    startCov=\n    {self.startCov},\n    startMean=\n    {self.startMean}"

    def filterStep(self,apostY0,apostErr0,obs):
        """Filtering step in Kalman algorithm

        Args:
            apostY0 (ndarray (d,) ): aposteriori prediction from the previous step
            apostErr0 (ndarray (d,d) ): its prediction error covariance
            obs (ndarray (k,) ): the current observation

        Returns:
            aprioriY ndarray (d,): current apriori prediction
            aprioriErr ndarray (d,d): current apriori prediction error covariance
            aposterioriY ndarray (d,): current aposteriori prediction
            aposterioriErr ndarray (d,d): current aposteriori prediction error covariance
            K ndarray (d,k): Kalman gain matrix
        """        
        aprioriY = self.A@apostY0 #new 
        aprioriErr = self.A@(apostErr0@(self.A.T)) + self.Ry #new apriori error
        
        K = aprioriErr@(self.B.T)@np.linalg.inv(self.Rx + self.B@aprioriErr@(self.B.T))# Kalman gain
        aposterioriY = aprioriY + K@(obs - self.B@aprioriY)
        aposterioriErr = aprioriErr - K@self.B@aprioriErr

        return aprioriY,aprioriErr,aposterioriY,aposterioriErr, K
    
    def filterSignal(self, signal, returnK=False):
        """Performs filtration pass

        Args:
            signal (ndarray (d,T) ): observations to filter, X
            returnK (bool, optional): whether to return Kalman gain matrix

        Returns:
            filteredSignal (ndarray (d,T)): filtered(or estimated) signal
            errs (ndarray (d,d,T)): filtered(or estimated) signal error covariance
            aprSignals (ndarray (d,T)): apriori predictions of the signal
            aprErrs (ndarray (d,d,T)): apriori predictions of the signal error covariance
            K (ndarray (d,K)): Kalman gain matrix (if returnK=True)
        """        

        filteredSignal = np.zeros([self.A.shape[0],signal.shape[-1]])
        aprSignals = np.zeros([self.A.shape[0],signal.shape[-1]])
        errs = np.zeros([self.A.shape[0],self.A.shape[0],signal.shape[-1]])
        aprErrs = np.zeros([self.A.shape[0],self.A.shape[0],signal.shape[-1]])
        filteredSignal[:,0] = self.startMean

        aprSig0 = filteredSignal[:,0]
        aprErr0 = self.startCov
        errs[:,:,0] = self.startCov
        aprErrs[:,:,0] = self.startCov
        aprSignals[:,0] = self.startMean

        for i in np.arange(1,signal.shape[1]):
            aprSig,aprErr,apostSig,apostErr, K = self.filterStep(aprSig0,aprErr0,signal[:,i])
            aprSig0 = apostSig
            aprErr0 = apostErr
            
            filteredSignal[:,i] = apostSig
            errs[:,:,i] = apostErr
            aprSignals[:,i] = aprSig
            aprErrs[:,:,i] = aprErr

        if(returnK):
            return filteredSignal, errs, aprSignals, aprErrs, K
        else:
            return filteredSignal, errs, aprSignals, aprErrs
    
    def smoothSignal(self, filteredSignal, errs, aprSignal, aprErrs, K=None, estLag1=False, returnGains=False):
        """Perfoms smoothing pass given filtering step results

        Args:
            filteredSignal (ndarray (d,T)): filtered(or estimated) signal
            errs (ndarray (d,d,T)): filtered(or estimated) signal error covariance
            aprSignal (ndarray (d,T)): apriori predictions of the signal
            aprErrs (ndarray (d,d,T)): apriori predictions of the signal error covariance
            returnGains (bool, optional): Whether to return smoothing gains

        Returns:
            smoothedSignal (ndarray (d,T)): smoothed signal
            smoothedErrs (ndarray (d,d,T)): smoothed signal error covariance
            gains (list of ndarray): smoothing gains
        """        
        #init est
        smoothedSignal = np.zeros_like(filteredSignal)
        smoothedSignal [...,-1] = filteredSignal[...,-1]
        
        #init cov
        smoothedErrs = np.zeros_like(errs)
        smoothedErrs[...,-1] = errs[...,-1]
        if(estLag1):
            lag1 = np.zeros_like(smoothedErrs)
            lag1 = lag1[...,:-1]
            lag1[...,-1] = (np.eye(smoothedErrs.shape[0]) - K@self.B)@self.A@errs[...,-2]
        if(returnGains):
            gains = []
        for t in np.arange(errs.shape[-1]-2,-1,-1):           
            smoothingGain = errs[...,t]@(np.transpose(self.A))@np.linalg.inv(aprErrs[...,t+1])
            if(returnGains):
                gains.append(smoothingGain)
            smoothedSignal[...,t] = filteredSignal[...,t] + smoothingGain@(smoothedSignal[...,t+1]-aprSignal[...,t+1])
            smoothedErrs[...,t] = errs[...,t] + \
                smoothingGain@(smoothedErrs[...,t+1] - aprErrs[...,t+1])@np.transpose(smoothingGain)
            if(estLag1):
                if(t<errs.shape[-1]-2):
                    lag1[...,t] =  errs[...,t]@smoothingGain.T + gains[-2]@(lag1[...,t+1] - self.A@errs[...,t+1])@smoothingGain.T
            
        if(returnGains):
            gains = gains[::-1]
            gains = np.concatenate([gain[:,:,None] for gain in gains], axis=-1)

            if(estLag1):
                return smoothedSignal, smoothedErrs, gains, lag1
            else:
                return smoothedSignal, smoothedErrs, gains
        else:
            if(estLag1):
                return smoothedSignal, smoothedErrs, lag1
            else:
                return smoothedSignal, smoothedErrs

    def fit(self, signal, Niter=1000, fixA=False, fixB=False, fixRx=False, fixRy=False, fixStartMean=False, fixStartCov=False):
        """Fits Kalman filter with EM-algorithm. Beware of divergence and carefully place hyperparameters.

        Args:
            signal (ndarray (d,T)): signal to filter, X
            Niter (int, optional): number of EM iterations. Defaults to 1000.
            fixA (bool, optional): Whether to fit A
            fixB (bool, optional): Whether to fit B
            fixRx (bool, optional): Whether to fit Rx
            fixRy (bool, optional): Whether to fit Ry
            fixStartMean (bool, optional): Whether to fit startMean
            fixStartCov (bool, optional): Whether to fit startCov
        """        
        for i in tqdm(np.arange(Niter)):
            filteredSignal, errs, aprSignal, aprErrs, K = self.filterSignal(signal,returnK=True)
            smoothedSignal, smoothedErrs, smoothGains, lag1 = self.smoothSignal(filteredSignal, errs, aprSignal, aprErrs, K=K, estLag1=True, returnGains=True)
            
            delta = np.einsum("it,jt -> ij",signal,smoothedSignal)
            gamma = np.einsum("it,jt -> ij",smoothedSignal,smoothedSignal) + np.sum(smoothedErrs,axis=-1)
            alpha = np.einsum("it,jt -> ij",signal,signal)
            beta =  np.einsum("it,jt -> ij",smoothedSignal[...,1:],smoothedSignal[...,:-1]) + np.sum(lag1,axis=-1)
            gam1 = gamma - smoothedSignal[:,None,-1]*smoothedSignal[None,:,-1] - smoothedErrs[...,-1]
            gam2 = gamma - smoothedSignal[:,None,0]*smoothedSignal[None,:,0] - smoothedErrs[...,0]
            
            if(not fixStartMean):
                self.startMean = smoothedSignal[...,0]
            if(not fixStartCov):
                self.startCov = smoothedErrs[...,0]

            if(not fixB):
                self.B = delta@np.linalg.inv(gamma)
                
            if(not fixRx):
                self.Rx = (alpha - self.B@delta.T)/signal.shape[-1]
            
            
            if(not fixA):
                self.A = beta@np.linalg.inv(gam1)
            
            if(not fixRy):
                self.Ry = (gam2-self.A@beta.T)/(signal.shape[-1]-1)
            
            #debug place
            if(i % 50 ==0):
                print(np.mean(smoothGains,axis=-1))
                print(np.mean(smoothedSignal,axis=-1))
            