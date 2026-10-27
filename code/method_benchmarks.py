"""Seven frozen, explicit normal-input reliability benchmark families.

All methods see only evaluate/event. Latent bases and reference integrals are
used exclusively by the benchmark evaluator. Original low-dimensional examples
and their orthogonal lifts count as one family, not separate examples.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm, chi2, ncx2


class Problem:
    def __init__(self, number, d=100):
        self.number=number; self.d=d if number!=6 else 20
        self.p={1:4,2:1,3:4,4:4,5:2,6:2,7:3}[number]
        self.r={1:4,2:2,3:2,4:5,5:2,6:self.d,7:2}[number]
        self.basis=np.linalg.qr(np.random.default_rng(9000+number).normal(size=(self.d,self.r)))[0]
        self.name={1:'Linear series',2:'Curved RP22',3:'Four branch',4:'Shared/private',
                   5:'Rare disks',6:'Full-rank radial',7:'Series-parallel RP57'}[number]

    def evaluate(self,u):
        z=u@self.basis; n=self.number
        if n==1:
            return 3.-z
        if n==2:
            return (2.5-(z[:,0]+z[:,1])/np.sqrt(2)+.1*(z[:,0]-z[:,1])**2)[:,None]
        if n==3:
            a,b=z.T
            return np.column_stack([3+.1*(a-b)**2-(a+b)/np.sqrt(2),
                                    3+.1*(a-b)**2+(a+b)/np.sqrt(2),
                                    a-b+7/np.sqrt(2),b-a+7/np.sqrt(2)])
        if n==4:
            c=np.array([.7,-.7,.5,-.5]); b=np.array([3.,3.1,3.2,3.3])
            return b-z[:,0,None]*c-.7*z[:,1:]-.15*(z[:,1:]**2-1)
        if n==5:
            return np.column_stack([((z-np.array([3.5,0.]))**2).sum(1)-.75**2,
                                    ((z-np.array([-4.1,.3]))**2).sum(1)-.55**2])
        if n==6:
            c=chi2.ppf(.997,self.d//2)
            return np.column_stack([c-(z[:,:self.d//2]**2).sum(1),
                                    c-(z[:,self.d//2:]**2).sum(1)])
        a,b=z.T
        return np.column_stack([-a*a+b**3+3,2-a-8*b,(a+3)**2+(b+3)**2-4])

    def event(self,g):
        if self.number==7:
            return ((g[:,0]<=0)&(g[:,1]<=0)) | (g[:,2]<=0)
        return (g<=0).any(1)

    def reference(self):
        if self.number==1:
            return float(1-ndtr(3.)**4),0.
        if self.number==2:
            return quad(lambda t: norm.sf(2.5+.2*t*t)*norm.pdf(t),-12,12,epsabs=1e-12)
        if self.number==3:
            v,e=quad(lambda t:2*norm.sf(3+.2*t*t)*norm.pdf(t),-3.5,3.5,epsabs=1e-12)
            return float(v+2*norm.sf(3.5)),e
        if self.number==4:
            c=np.array([.7,-.7,.5,-.5]); b=np.array([3.,3.1,3.2,3.3])
            def f(t):
                disc=.7**2-4*.15*(c*t-b-.15)
                ds=np.sqrt(np.maximum(disc,0))
                prob=np.where(disc>0,ndtr((-.7+ds)/.3)-ndtr((-.7-ds)/.3),0.)
                return (1-prob.prod())*norm.pdf(t)
            return quad(f,-12,12,epsabs=1e-11,limit=200)
        if self.number==5:
            return float(ncx2.cdf(.75**2,2,3.5**2)+ncx2.cdf(.55**2,2,4.1**2+.3**2)),0.
        if self.number==6:
            return 1-.997**2,0.
        def conditional(t):
            intervals=[]; a=2-8*t; v=t**3+3
            if v<=0:
                intervals.append((a,np.inf))
            else:
                s=np.sqrt(v)
                if a<=-s: intervals.append((a,-s))
                intervals.append((max(a,s),np.inf))
            if abs(t+3)<2:
                s=np.sqrt(4-(t+3)**2); intervals.append((-3-s,-3+s))
            merged=[]
            for lo,hi in sorted(intervals):
                if merged and lo<=merged[-1][1]: merged[-1][1]=max(hi,merged[-1][1])
                else: merged.append([lo,hi])
            return sum(ndtr(hi)-ndtr(lo) for lo,hi in merged)*norm.pdf(t)
        return quad(conditional,-12,12,points=[-5,-1,-np.cbrt(3)],epsabs=1e-10,limit=500)
