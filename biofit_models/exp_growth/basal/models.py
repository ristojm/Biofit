def Inverse_Monod(S,mumax,Ks,a):
    return a+((Ks*S)/(mumax-S))
def exp_growth(S,r,a):
    return a*((1+r)**S)
