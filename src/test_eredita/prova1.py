class Equation():
    def __init__(self,par):
        self.name = "Equazione" + par
    def pre_process(self): pass
    def post_process(self): pass
    def compute_residual(self): pass

class CoreNN():
    def __init__(self,par):
        self.par = par
        self.model = "modello"

class LossNN(CoreNN):
    def __init__(self,b,**kw):
        self.b=b
        super(LossNN,self).__init__(**kw)

class PredNN(CoreNN):
    def __init__(self,c,d,**kw):
        self.c = c
        self.d = d
        super(PredNN,self).__init__(**kw)

class BayesNN(LossNN,PredNN):
    def __init__(self,par):
        equation = Equation(par)
        b = equation.compute_residual
        c = equation.pre_process
        d = equation.post_process
        super(BayesNN,self).__init__(par=par,b=b,c=c,d=d)

bnn = BayesNN("Parametri")
print("*** MAIN ***")
print("model =", bnn.model)
print("par =", bnn.par)
print("b =", bnn.b)
print("c =", bnn.c)
print("d =", bnn.d)

"""
1. LossNN, PredNN, BayesNN contengono model e par grazie a CoreNN
2. BayesNN inizializza l'eqauzione e passa 
    - compute_residual a LossNN
    - pre e post process a PredNN
    successivamente "dimentica" l'equazione lasciandola separata nelle figlie
"""