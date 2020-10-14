import autograd.numpy as np
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
