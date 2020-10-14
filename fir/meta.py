import autograd.numpy as np

class Sequential:
    def __init__(self,layers):
        self.layers = layers

        self.n_params=0
        for l in self.layers:
            self.n_params+=l.n_params

        self.n_inp = self.layers[0].n_inp
        self.n_outp = self.layers[-1].n_out
    def forward(self,X):
        h=X
        for i in range(0,len(self.layers)):
            h=self.layers[i].forward(h)
        return h
    def get_regularization(self):
        sum1=0.0
        for i in range(0,len(self.layers)):
            sum1+=self.layers[i].get_regularization()
        return sum1
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
    
class ParallelConcat:
    def __init__(self,layers):
        self.layers = layers
        self.n_params = np.sum([l.n_params for l in self.layers])

    def forward(self,X):
        assert(len(X)==len(self.layers))
        outps = [self.layers[i].forward(X[i]) for i in range(0,len(self.layers))]
        return np.concatenate(outps,axis=-1)
    def get_regularization(self):
        return np.sum([l.get_regularization() for l in self.layers])
    def get_params(self):
        return np.concatenate([l.get_params() for l in self.layers])
    def set_params(self,ps):
        n=0
        for l in self.layers:
            m=l.n_params
            l.set_params(ps[n:n+m])
            n+=m

class ParallelSum:
    def __init__(self,layers):
        self.layers = layers
        self.n_params = np.sum([l.n_params for l in self.layers])

    def forward(self,X):
        assert(len(X)==len(self.layers))
        outps = [self.layers[i].forward(X[i]) for i in range(0,len(self.layers))]
        tmp = 0.0
        for o in outps:
            tmp+=o
        return tmp

    def get_regularization(self):
        return np.sum([l.get_regularization() for l in self.layers])
    def get_params(self):
        return np.concatenate([l.get_params() for l in self.layers])
    def set_params(self,ps):
        n=0
        for l in self.layers:
            m=l.n_params
            l.set_params(ps[n:n+m])
            n+=m    
