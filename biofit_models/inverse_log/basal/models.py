import numpy as np

def Menten(S, mu_max, Ks,a):
    return a+(1-((mu_max*S)/(Ks+S)))

def Han(S, mu_max, Ks, S_max, n, m,a):
    return a+(1-((mu_max*S*((1-(S/S_max))**n))/(S+Ks*((1-(S/S_max))**m))))

def Luong(S, mu_max, Ks, S_max, n,a):
    return a+(1-((mu_max*S*((1-(S/S_max))**n))/(S+Ks)))

def Andrews(S,mu_max,Ks,Ki,a):
    return a+(1-(mu_max*S/((Ks+S)*(1+(S/Ki)))))

def Aiba(S,mu_max,Ks,Ki,a):
    return a+(1-((mu_max*S/(Ks+S))*np.exp(-S/Ki)))

def Moser(S,mu_max,Ks,n,a):
    return a + (mu_max*(1-(1/(1+((Ks*S)**-n)))))
    #return a+(1-(mu_max*(S**n)/((Ks**n)+(S**n))))

def Edward(S,mu_max,Ks,Ki,a):
    return a+(1-(mu_max*S*(np.exp(-S/Ki)-np.exp(-S/Ks))))

def Webb(S,mu_max,Ks,Ki,K,a):
    return a+(1-((mu_max*S*(1+(S/K)))/(S+Ks+((S**2)/Ki))))

def Yano(S,mu_max,Ks,Ki,K,a):
    return a+(1-(mu_max*S/(Ks+S+(((S**2)/Ki)*(1+(S/K))))))

def Haldane(S,mu_max,Ks,S_max,Ki,a):
    return a+(1-(mu_max*S/(Ks+S+(S_max/Ki))))
