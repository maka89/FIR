import autograd.numpy as np

from autograd import grad
from scipy.optimize import fmin_l_bfgs_b

class Model:
    def __init__(self, layers):
        self.layers = layers
        self.errf_grad = grad(self.errf,0)
    def forward(self,X):
        h=X
        for i in range(0,len(self.layers)):
            h=self.layers[i].forward(h)
        return h
    def get_params(self):
        ps = []
        for i in range(0,len(self.layers)):
            ps.append(self.layers[i].get_params())
        return np.concatenate(ps)
    def set_params(self,ps):
        n=0
        for i in range(0,len(self.layers)):
            m=self.layers[i].n_params
            self.layers[i].set_params(ps[n:n+m])
            n+=m
    def errf(self,ps,args):
        X,Y = args[0],args[1]
        self.set_params(ps)
        Yp = self.forward(X)

        err = np.sum((Y-Yp)**2)
        for i in range(0,len(self.layers)):
            err += self.layers[i].get_regularization()
        return err
    def fit(self,X,Y):

        ps0 = self.get_params()
        x,f,d = fmin_l_bfgs_b(self.errf,ps0,fprime=self.errf_grad,args=([X,Y],),factr=10,pgtol=1e-10)
    def predict(self,X):
        return self.forward(X)
