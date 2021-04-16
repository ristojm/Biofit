import numpy as np

def Menten(S, mu_max, Ks):
    return ((mu_max*S)/(Ks+S))

def Han(S, mu_max, Ks, S_max, n, m):
    return (mu_max*S*((1-(S/S_max))**n))/(S+Ks*((1-(S/S_max))**m))

def Luong(S, mu_max, Ks, S_max, n):
    return (mu_max*S*((1-(S/S_max))**n))/(S+Ks)

def Andrews(S,mu_max,Ks,Ki):
    return mu_max*S/((Ks+S)*(1+(S/Ki)))

def Aiba(S,mu_max,Ks,Ki):
    return (mu_max*S/(Ks+S))*np.exp(-S/Ki)

def Moser(S,mu_max,Ks,n):
    return (mu_max*(S**n))/(Ks+(S**n))

def Edward(S,mu_max,Ks,Ki):
    return mu_max*S*(np.exp(-S/Ki)-np.exp(-S/Ks))

def Webb(S,mu_max,Ks,Ki,K):
    return (mu_max*S*(1+(S/K)))/(S+Ks+((S**2)/Ki))

def Yano(S,mu_max,Ks,Ki,K):
    return (mu_max*S)/(Ks+S+(((S**2)/Ki)*(1+(S/K))))

def Haldane(S,mu_max,Ks,S_max,Ki):
    return (mu_max*S)/(Ks+S+(S_max/Ki))
