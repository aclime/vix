

import math
import numpy as np
from scipy.special import erf
from scipy import stats
norm = stats.norm

class BSOPM_Class:
  
  def disc_function(self, FV, r, T):
    PV = FV * np.exp(-r*T)
    return PV

  def bs_d1_d2(self,St,r,t,K,call,sig):
    d1 = np.log(St/K)
    d1 += ( sig*sig/2 + r)*t
    with np.errstate(divide='ignore'):
        d1/=sig * t**0.5
    d2=d1-sig * t**0.5
    return d1,d2

  def cdf_approx(self,dn,call):
    if call:
      Ndn = (0.50 * (1.0 + erf(dn / math.sqrt(2.0))))
    else:
      Ndn = (0.50 * (1.0 + erf(-dn / math.sqrt(2.0))))
    return Ndn
          
  def bs_delta(self,d1,d2,call):
    Nd1 = self.cdf_approx(dn=d1,call=call)
    Nd2 = self.cdf_approx(dn=d2,call=call)
    return Nd1,Nd2

  def bs_gamma(self,d1,St,sig,t):
    gamma = norm.pdf(d1)
    with np.errstate(divide='ignore'):
        gamma /= (St*sig*np.sqrt(t)) 
    return gamma

  def bs_price(self,St,r,t,K,call,Nd1,Nd2,T):
    pvk = self.disc_function(K,r, T-t)
    if call:
      price = St*Nd1-pvk*Nd2
    else:
      price = pvk * Nd2 - St * Nd1 
    return price
  
  def opt_payoff(self, ST, K, call=True):
    if call == True:
      payoff=np.maximum(ST-K,0)
    else:
      payoff=np.maximum(K-ST,0)
    return payoff

  def __init__(self,S0,r,sigma,t,T,K,call=True):
    self.S0 = S0
    self.r = r
    self.sigma = sigma
    self.T  = T
    self.K = K
    self.call = call
    self.t = t
        
    self.d1,self.d2=self.bs_d1_d2(St=self.S0,r=self.r,t=self.T-self.t,K=self.K,call=self.call,sig=self.sigma)
    self.Nd1,self.Nd2=self.bs_delta(d1=self.d1,d2=self.d2,call=self.call)
    self.delta=self.Nd1
    self.gammas = self.bs_gamma(d1=self.d1,St=self.S0,sig=self.sigma,t=self.T-self.t)
    self.price = self.bs_price(St=self.S0,r=self.r,t=self.t,K=self.K,call=self.call,Nd1=self.Nd1,Nd2=self.Nd2,T=self.T)
    self.payoff = self.opt_payoff(self.S0,self.K,self.call)
    