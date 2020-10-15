import autograd.numpy as np
import math


class FIR:
    def __init__(self,n_inp,n_out,fir_length):
        self.n_inp = n_inp
        self.n_out = n_out
        self.fir_length = fir_length
        self.n_params = n_inp*n_out*fir_length+n_out
        self.initialize()
    def initialize(self):
        self.W = np.random.randn(self.fir_length,self.n_inp,self.n_out)/np.sqrt(self.n_inp*self.fir_length)
        self.b = np.zeros(self.n_out)

    def set_params(self,ps):
        n = self.n_inp*self.n_out*self.fir_length
        self.W = ps[0:n].reshape(self.fir_length,self.n_inp,self.n_out)
        self.b = ps[n:n+self.n_out]
    def get_params(self):
        return np.concatenate((self.W.ravel(),self.b.ravel()))
    def get_regularization(self):
        return 0.0
    def forward(self,x):
        # X - [n_ex, length, n_in]
        # Y - [n_ex, length, n_out]

        #Y=np.zeros((X.shape[0],X.shape[1],self.n_out))
        
        lf = self.fir_length
        lx = x.shape[1]

        x2 = np.concatenate((x,np.zeros((x.shape[0],lf,x.shape[2]))),axis=1)
        f2 = np.concatenate((self.W, np.zeros((lx,self.n_inp,self.n_out))),axis=0)
        X = np.fft.fft(x2,axis=1)
        F = np.fft.fft(f2,axis=0)


        tmp = np.sum(F[np.newaxis,...]*X[...,np.newaxis],axis=2)
        Y = np.real(np.fft.ifft(tmp,axis=1))[:,0:x.shape[1]]
        Y+= self.b.reshape(1,1,-1)

        return Y
class FIRLP:
    def __init__(self,n_inp,n_out,fir_length,n_freqs):
        self.n_inp = n_inp
        self.n_out = n_out
        self.fir_length = fir_length
        self.n_freqs=n_freqs
        self.n_params = n_inp*n_out*(1+2*n_freqs)+n_out
        self.initialize()
    def initialize(self):
        self.FWr = np.random.randn(self.n_freqs,self.n_inp,self.n_out)/np.sqrt(self.n_inp*self.n_freqs)
        self.FWi = np.random.randn(self.n_freqs,self.n_inp,self.n_out)/np.sqrt(self.n_inp*self.n_freqs)
        self.FWc = np.random.randn(self.n_inp,self.n_out)/np.sqrt(self.n_inp)
        self.b = np.zeros(self.n_out)

    def set_params(self,ps):
        n=0
        m = self.n_inp*self.n_out*self.n_freqs
        self.FWr = ps[n:n+m].reshape(self.n_freqs,self.n_inp,self.n_out)
        n+=m

        m=self.n_inp*self.n_out*self.n_freqs
        self.FWi = ps[n:n+m].reshape(self.n_freqs,self.n_inp,self.n_out)
        n+=m

        m=self.n_inp*self.n_out
        self.FWc = ps[n:n+m].reshape(self.n_inp,self.n_out)
        n+=m

        m=self.n_out
        self.b = ps[n:n+m]
    def get_params(self):
        return np.concatenate((self.FWr.ravel(),self.FWi.ravel(),self.FWc.ravel(),self.b.ravel()))
    def get_regularization(self):
        return 0.0
    def get_W(self):

        n=self.fir_length-1-2*self.n_freqs
        FW1 = self.FWr+1j*self.FWi

        ll = [self.FWc.reshape(1,self.n_inp,self.n_out)+0j,FW1,np.zeros((n,self.n_inp,self.n_out))+0j,np.conj(FW1[::-1])]
        FW  = np.concatenate(ll,axis=0)
        #print(FW.dtype)
        return np.real(np.fft.ifft(FW,axis=0))

    def forward(self,x):
        # X - [n_ex, length, n_in]
        # Y - [n_ex, length, n_out]

        #Y=np.zeros((X.shape[0],X.shape[1],self.n_out))
        

        W=self.get_W()
        lf = self.fir_length
        lx = x.shape[1]

        x2 = np.concatenate((x,np.zeros((x.shape[0],lf,x.shape[2]))),axis=1)
        f2 = np.concatenate((W, np.zeros((lx,self.n_inp,self.n_out))),axis=0)
        X = np.fft.fft(x2,axis=1)
        F = np.fft.fft(f2,axis=0)


        tmp = np.sum(F[np.newaxis,...]*X[...,np.newaxis],axis=2)
        Y = np.real(np.fft.ifft(tmp,axis=1))[:,0:x.shape[1]]
        Y+= self.b.reshape(1,1,-1)

        return Y

class FIR_L2(FIR):
    def __init__(self,n_inp,n_out,fir_length,l2):
        super().__init__(n_inp,n_out,fir_length)
        self.l2 = l2
    def get_regularization(self):
        return np.sum(self.l2*self.W**2)
    
class FIR_L2FFT(FIR):
    def __init__(self,n_inp,n_out,fir_length,l2,cutoff,cutoff_slope,alpha):
        super().__init__(n_inp,n_out,fir_length)
        self.l2 = l2
        self.cutoff = cutoff
        self.cutoff_slope=cutoff_slope
        self.alpha=alpha

    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    def get_regularization(self):
        #L2
        tmp = self.l2*np.sum(self.W**2)

        #Input highfreq-reg
        fft_w = np.fft.fft(self.W,axis=0)
        freqs = np.fft.fftfreq(self.W.shape[0], 0.5/self.W.shape[0])
        afreqs = np.abs(freqs)
        tmp2 = self.alpha*np.sum(self.sigmoid( self.cutoff_slope*(afreqs-self.cutoff))*np.sum(np.abs(fft_w)**2,axis=(1,2)))
        return tmp + tmp2

class FIR_L2FFT2(FIR):
    def __init__(self,n_inp,n_out,fir_length,l2,cutoff,alpha):
        super().__init__(n_inp,n_out,fir_length)
        self.l2 = l2
        self.cutoff = cutoff
        self.alpha=alpha

    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    def get_regularization(self):
        #L2
        tmp = self.l2*np.sum(self.W**2)

        #Input highfreq-reg
        fft_w = np.fft.fft(self.W,axis=0)
        freqs = np.fft.fftfreq(self.W.shape[0], 0.5/self.W.shape[0])
        afreqs = np.abs(freqs)
        tmp2 = self.alpha*np.sum(np.abs(fft_w[afreqs >= self.cutoff,:,:])**2)
        return tmp + tmp2
class Dense:
    def __init__(self,n_inp,n_out):
        self.n_inp=n_inp
        self.n_out=n_out
        self.n_params = n_inp*n_out+n_out
        self.initialize()

    
    def set_params(self,ps):
        n=self.n_inp*self.n_out
        self.W=ps[0:n].reshape(self.n_inp,self.n_out)
        self.b=ps[n:n+self.n_out]

    def initialize(self):
        self.W=np.random.randn(self.n_inp,self.n_out)/np.sqrt(self.n_inp)
        self.b=np.zeros(self.n_out)

    def forward(self,X):
        assert(X.shape[1]==self.W.shape[0])
        return np.dot(X,self.W)+self.b

    def get_params(self):
        return np.concatenate((self.W.ravel(),self.b.ravel()))

    def get_regularization(self):
        return 0.0

class DenseL2(Dense):
    def __init__(self,n_inp,n_out,l2):
        super().__init__(n_inp,n_out)
        self.l2 = l2
    def get_regularization(self):
        return np.sum(self.l2*self.W**2)

class DenseL2FFT(Dense):
    def __init__(self,n_inp,n_out,l2,cutoff,cutoff_slope,alpha):
        super().__init__(n_inp,n_out)
        self.l2 = l2
        self.cutoff=cutoff
        self.cutoff_slope=cutoff_slope
        self.alpha=alpha
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    def get_regularization(self):
        #L2
        tmp = self.l2*np.sum(self.W**2)

        #Input highfreq-reg
        fft_w = np.fft.fft(self.W,axis=0)
        freqs = np.fft.fftfreq(self.W.shape[0], 0.5/self.W.shape[0])
        afreqs = np.abs(freqs)
        tmp2 = self.alpha*np.sum(self.sigmoid( self.cutoff_slope*(afreqs-self.cutoff))*np.sum(np.abs(fft_w)**2,axis=1))
        return tmp + tmp2

class Activation:
    def __init__(self,atype='relu'):
        if atype == 'relu':
            self.fn= lambda x: (x>0)*x
        elif atype == 'tanh':
            self.fn= lambda x: np.tanh(x)
        elif atype == 'id':
            self.fn=lambda x:x
        else:
            self.fn=lambda x:x
        self.n_params=0
        self.get_regularization = lambda : 0.0

    def initialize(self):
        return
    def set_params(self,ps):
        return
    def get_params(self):
        return np.zeros(0)
    def forward(self,x):
        return self.fn(x)




    