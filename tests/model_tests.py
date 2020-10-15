import fir
from fir import Model,Dense,DenseL2,DenseL2FFT,Activation,ParallelSum,FIR,FIR_L2FFT,FIR_L2,Sequential,FIRLP_L2,FIR_L2FFT2,Conv1D,Conv1D_L2
from sklearn.linear_model import LinearRegression,Ridge
import numpy as np




def test_1():
    X = np.random.randn(100,17)
    Y = np.random.randn(100,10)


    l2 = 0.0
    lr= Ridge(l2)
    lr.fit(X,Y)

    dft_r = Model([Dense(17,20),Activation('id'),Dense(20,10)] )
    dft_r.fit(X,Y)

    Xp=np.random.randn(1000,17)

    Yp1=lr.predict(Xp)
    Yp2=dft_r.predict(Xp)

    if np.max(np.abs(Yp1-Yp2)) <= 1e-6:
        return True
    else:
        return False

def test_2():
    X = np.random.randn(100,17)
    Y = np.random.randn(100,10)


    l2 = 1.0
    lr= Ridge(l2)
    lr.fit(X,Y)

    dft_r = Model([DenseL2(17,10,l2)] )
    dft_r.fit(X,Y)

    Xp=np.random.randn(1000,17)

    Yp1=lr.predict(Xp)
    Yp2=dft_r.predict(Xp)

    if np.max(np.abs(Yp1-Yp2)) <= 1e-6:
        return True
    else:
        return False

def test_3():
    X = np.random.randn(100,17)
    Y = np.random.randn(100,10)


    l2 = 1.0
    lr= Ridge(l2)
    lr.fit(X,Y)

    dft_r = Model([DenseL2FFT(17,10,l2,10.0,10.0,0.0)] )
    dft_r.fit(X,Y)

    Xp=np.random.randn(1000,17)

    Yp1=lr.predict(Xp)
    Yp2=dft_r.predict(Xp)

    if np.max(np.abs(Yp1-Yp2)) <= 1e-6:
        return True
    else:
        return False
def test_4():
    X = np.random.randn(100,17)
    Y = np.random.randn(100,10)


    l2 = 1.0
    lr= Ridge(l2)
    lr.fit(X,Y)



    dft_r = Model([ParallelSum([DenseL2(10,10,l2),DenseL2(7,10,l2)] )])
    dft_r.fit([X[:,0:10],X[:,10::]],Y)

    Xp=np.random.randn(1000,17)

    Yp1=lr.predict(Xp)
    Yp2=dft_r.predict([Xp[:,0:10],Xp[:,10::]])

    if np.max(np.abs(Yp1-Yp2)) <= 1e-6:
        return True
    else:
        return False
    
def convolve(fir,x):
    x2 = np.concatenate((x,np.zeros_like(fir)))
    f2 = np.concatenate((fir,np.zeros_like(x)))
    X = np.fft.fft(x2)
    F = np.fft.fft(f2)
    res = np.real(np.fft.ifft(X*F)[0:len(x)])
    return res
def test_5():
    X = np.random.randn(3,1000,1)
    Y = np.zeros_like(X)
    fir = np.random.randn(30)
    b= 1.336
    for i in range(0,X.shape[0]):
        Y[i,:,0]= convolve(fir,X[i,:,0])+b
    
    dft_r = Model([FIR(1,1,len(fir))])
    dft_r.fit(X,Y)

    f1 = fir
    f2 = dft_r.layers[0].W[:,0,0]
    if np.max(np.abs(f1-f2)) <= 1e-6 and np.abs(b-dft_r.layers[0].b) <= 1e-6:
        return True
    else:
        return False
    
def test_6():
    n_in = 3
    n_out = 5
    n_ex = 10
    xlen = 500
    fir_len = 30


    X = np.random.randn(n_ex,xlen,n_in)
    Y = np.zeros((n_ex,xlen,n_out))
    firs = np.random.randn(fir_len,n_in,n_out)
    bs = np.random.randn(n_out)

    for i in range(0,n_ex):
        for j in range(0,n_in):
            for k in range(0,n_out):
                Y[i,:,k] += convolve(firs[:,j,k],X[i,:,j])
    
    for i in range(0,n_ex):
        for k in range(0,n_out):
            Y[i,:,k]+=bs[k]



    dft_r = Model([FIR(n_in,n_out,fir_len)])
    dft_r.fit(X,Y)

    f1 = firs
    f2 = dft_r.layers[0].W

    if np.max(np.abs(f1-f2)) <= 1e-8 and np.max(np.abs(bs-dft_r.layers[0].b)) <= 1e-8:
        return True
    else:
        return False

def test_7():
    n_in = 3
    n_out = 5
    n_ex = 10
    xlen = 500
    fir_len = 30


    X = np.random.randn(n_ex,xlen,n_in)
    Y = np.zeros((n_ex,xlen,n_out))
    firs = np.random.randn(fir_len,n_in,n_out)
    bs = np.random.randn(n_out)

    for i in range(0,n_ex):
        for j in range(0,n_in):
            for k in range(0,n_out):
                Y[i,:,k] += convolve(firs[:,j,k],X[i,:,j])
    
    for i in range(0,n_ex):
        for k in range(0,n_out):
            Y[i,:,k]+=bs[k]



    dft_r = Model([FIR_L2FFT(n_in,n_out,fir_len,0.0,30.2,100.0,1.0)])
    #dft_r = Model([FIR_L2(n_in,n_out,fir_len,0.0)])
    dft_r.fit(X,Y)

    f1 = firs
    f2 = dft_r.layers[0].W

    if np.max(np.abs(f1-f2)) <= 1e-6 and np.max(np.abs(bs-dft_r.layers[0].b)) <= 1e-6:
        return True
    else:
        return False

def test_8():


    n_ex=1000
    n_in=10
    h=50
    n_out=20
    X = np.random.randn(n_ex,n_in)
    Y = np.random.randn(n_ex,n_out)
    l2_1=1.0
    l2_2=1.3


    m1 = Model([DenseL2(n_in,h,l2_1),DenseL2(h,n_out,l2_2)])
    m1.fit(X,Y)

    m2 = Model([Sequential([DenseL2(n_in,h,l2_1),DenseL2(h,n_out,l2_2)])])
    m2.fit(X,Y)

    m3 = Model([Sequential([DenseL2(n_in,h,l2_1)]),DenseL2(h,n_out,l2_2)])
    m3.fit(X,Y)
    

    Xp = np.random.randn(100,n_in)
    yp1 = m1.predict(Xp)
    yp2 = m2.predict(Xp)
    yp3 = m3.predict(Xp)

    if np.max(np.abs(yp1-yp2)) <= 1e-6 and np.max(np.abs(yp1-yp3)) <= 1e-6:
        return True
    else:
        return False

def test_9():


    n_ex=2
    xlen = 100
    n_in=4
    n_out=3
    fir_length=20
    l2 = 1.0
    X = np.random.randn(n_ex,xlen,n_in)
    Y = np.random.randn(n_ex,xlen,n_out)


    m1 = Model([FIRLP_L2(n_in,n_out,fir_length,2,l2)])
    m1.fit(X,Y)

    m2 = Model([FIR_L2FFT2(n_in,n_out,fir_length,l2,6,1e6)])
    m2.fit(X,Y)

    W1=m1.layers[0].get_W()
    W2=m2.layers[0].W


    if np.max(np.abs(W1-W2)) <= 1e-4:
        return True
    else:
        return False


def test_10():


    n_ex=5
    xlen = 100
    n_in=5
    n_out=3
    fir_length=7
    l2 = 1.0
    X = np.random.randn(n_ex,xlen,n_in)
    Y = np.random.randn(n_ex,xlen,n_out)


    m1 = Model([FIR_L2(n_in,n_out,fir_length,l2)])
    m1.fit(X,Y,disp=False)

    m2 = Model([Conv1D_L2(n_in,n_out,fir_length,l2)])
    m2.fit(X,Y,disp=False)

    W1 = m1.layers[0].W
    W2 = m2.layers[0].W
    if np.max(np.abs(W1-W2)) <= 1e-6:
        return True
    else:
        return False