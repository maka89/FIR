import autograd.numpy as np

from autograd import grad,primitive
from autograd.extend import defvjp
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import cholesky, solve_triangular
import time
@primitive
def cho_inv(x):
    L = cholesky(x, lower=True)
    L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
    K_inv = L_inv.dot(L_inv.T)
    return K_inv

def T(x): return np.swapaxes(x, -1, -2)
def grad_inv(ans, x):
    return lambda g: -np.dot(np.dot(T(ans), g), T(ans))
defvjp(cho_inv, grad_inv)

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
        X,Y,drop_first = args[0],args[1],args[2]
        self.set_params(ps)
        Yp = self.forward(X)

        if drop_first is not None:
            err = np.sum((Y[:,drop_first::,:]-Yp[:,drop_first::,:])**2)
        else:
            err = np.sum((Y-Yp)**2)
            
        for i in range(0,len(self.layers)):
            err += self.layers[i].get_regularization()
        return err
    def fit(self,X,Y,disp=False,maxiter=1000,pgtol = 1e-5,drop_first=None):

        ps0 = self.get_params()
        x,f,d = fmin_l_bfgs_b(self.errf,ps0,fprime=self.errf_grad,args=([X,Y,drop_first],),factr=10,pgtol=pgtol,disp=disp,maxiter=maxiter)

    def fit_adam(self,X,Y,disp=False,n_epochs=10,batch_size = 8,drop_first=None,beta_1=0.9,beta_2=0.999,learning_rate = 1e-3,disp_time = 3):
        
        
        m=int(X.shape[0]/batch_size)
        ps = self.get_params()
        mt=np.zeros_like(ps)
        vt=np.zeros_like(ps)
        args=[X,Y,drop_first]
        
        loss=1e3
        
        
        time0 = time.time()
        t=0
        
        for i in range(0,n_epochs):
            for j in range(0,m):
                
                bx = X[j*batch_size:(j+1)*batch_size]
                by = Y[j*batch_size:(j+1)*batch_size]
                args=[bx,by,drop_first]
                t=t+1
                ps = self.get_params()
                g = self.errf_grad(ps,args)
                mt = beta_1*mt + (1.0-beta_1)*g
                vt = beta_2*vt + (1.0-beta_2)*g**2

                amt = mt/(1.0-beta_1**t)
                avt = vt/(1.0-beta_2**t)
                ps_new = ps - learning_rate*amt/(np.sqrt(avt)+1e-8)
                self.set_params(ps_new)
                if time.time() - time0 > disp_time:
                    loss = self.errf(ps_new,[bx,by,drop_first])
                    print("epoch",i,"batch",j,"loss",loss)
                    time0 = time.time()
            args=[X,Y,drop_first]  
         
            loss = self.errf(ps_new,args)
            print("epoch",i,"loss",loss)

            

    def predict(self,X):
        return self.forward(X)


class LOOModel(Model):
    def errf_lstsq(self,ps,args):
        X,Y = args[0],args[1]
        self.set_params(ps)
        h0 = self.forward(X)
        sh = list(h0.shape)
        sh[-1]=1
        h = np.concatenate((h0,np.ones(sh)),axis=-1)

        hth = np.dot(h.T,h)
        hth = hth + np.eye(h.shape[1])*self.l2
        #hthi = np.linalg.inv(hth)
        hthi = cho_inv(hth)
        v=np.dot(h.T,Y)
        v=np.dot(hthi,v)
        yp=np.dot(h,v)

        err = np.sum((yp-Y)**2)
        for i in range(0,len(self.layers)):
            err += self.layers[i].get_regularization()
        
        return err

    def errf_loo(self,ps,args):
        X,Y = args[0],args[1]
        self.set_params(ps)
        h0 = self.forward(X)
        sh = list(h0.shape)
        sh[-1]=1
        h = np.concatenate((h0,np.ones(sh)),axis=-1)

        hth = np.dot(h.T,h)
        hth = hth + np.eye(h.shape[1])*self.l2
        hthi = cho_inv(hth)

        v=np.dot(h.T,Y)
        v=np.dot(hthi,v)
        yp=np.dot(h,v)
        hth2 = np.dot(hthi,h.T)
        betas = []
        for i in range(0,Y.shape[0]):
            betas.append(np.sum(hth2[:,i]*h[i,:]))
        betas = np.array(betas)

        err = np.sum(((yp-Y)/(1.0-betas))**2)
        for i in range(0,len(self.layers)):
            err += self.layers[i].get_regularization()
        return err

    def predict(self,X):
        h0 = self.forward(self.X)
        sh = list(h0.shape)
        sh[-1]=1
        h = np.concatenate((h0,np.ones(sh)),axis=-1)

        hth = np.dot(h.T,h)
        hth = hth + np.eye(h.shape[1])*self.l2
        hthi = cho_inv(hth)

        v=np.dot(h.T,self.Y)
        v=np.dot(hthi,v)

        h0 = self.forward(X)
        sh = list(h0.shape)
        sh[-1]=1
        h2 = np.concatenate((h0,np.ones(sh)),axis=-1)
        yp=np.dot(h2,v)
        return yp
    def __init__(self,layers,method="lstsq",l2=1e-8):
        super().__init__(layers)
        self.l2=l2
        if method=="lstsq":
            self.errf = self.errf_lstsq
            self.errf_grad = grad(self.errf,0)
        elif method == "loo":
            self.errf = self.errf_loo
            self.errf_grad = grad(self.errf,0)
        
    def fit(self,X,Y,disp=False,maxiter=1000,pgtol = 1e-5):
        self.X,self.Y=X,Y
        ps0 = self.get_params()
        x,f,d = fmin_l_bfgs_b(self.errf,ps0,fprime=self.errf_grad,args=([X,Y],),factr=10,pgtol=pgtol,disp=disp,maxiter=maxiter)
