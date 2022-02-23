def Inverse_Monod(S,mumax,Ks):
    return (Ks*S)/(mumax-S)

def exp_growth(S,r,Ks):
    return (1+r)**S
