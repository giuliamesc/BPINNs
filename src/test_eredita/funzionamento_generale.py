class CoreNN():
    def __init__(self,a):
        self.a=a

class LossNN(CoreNN):
    def __init__(self,b,**kw):
        self.b=b
        super(LossNN,self).__init__(**kw)

class PredNN(CoreNN):
    def __init__(self,c,**kw):
        self.c=c
        super(PredNN,self).__init__(**kw)

class BayesNN(LossNN,PredNN):
    def __init__(self,a,b,c,d):
        super(BayesNN,self).__init__(a=a,b=b,c=c)
        self.d=d

bnn = BayesNN(1,2,3,4)
print("a = ", bnn.a)
print("b = ", bnn.b)
print("c = ", bnn.c)
print("d = ", bnn.d)