"""Fig.2 visual-inspection replica of the advantage-hierarchy figure, but with
I=10 compositions instead of I=1. Same constructions (dp_accounting PLD,
exact Neyman-Pearson ROC, Dong-Roth-Su envelope). Standalone matplotlib PNG;
does not touch the paper."""
import sys, math
sys.path.insert(0, '.')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from dp_accounting import pld as dp_pld

EPS0, DELTA0 = 10.0, 1e-12
STEPS = 10
DISC = 1e-4
MECHS = [0.001, 0.01, 0.1, 0.3, 0.6, 1.0]
BLUES = ['#BDD7E7','#9ECAE1','#6BAED6','#4292C6','#2171B5','#08306B']
ORANGE = '#FD8C3B'

def mk(sigma,p): return dp_pld.privacy_loss_distribution.from_gaussian_mechanism(
    standard_deviation=sigma, sampling_prob=p, value_discretization_interval=DISC,
    use_connect_dots=True).self_compose(STEPS)
def calib(p):
    lo,hi=0.02,200.0
    for _ in range(90):
        m=0.5*(lo+hi)
        if mk(m,p).get_delta_for_epsilon(EPS0)>DELTA0: lo=m
        else: hi=m
        if hi-lo<1e-7*hi: break
    return 0.5*(lo+hi)
def roc(obj):
    pmf=obj._pmf_add; disc=pmf._discretization; lower=pmf._lower_loss
    q=np.asarray(pmf._probs,float); inf=float(pmf._infinity_mass); n=len(q)
    losses=(lower+np.arange(n))*disc
    pd=q[::-1]; ld=losses[::-1]; qd=np.exp(-ld)*pd
    Pgt=np.concatenate([[inf],np.cumsum(pd)+inf]); Qgt=np.concatenate([[0.],np.cumsum(qd)])
    adv=float(np.max(Pgt-Qgt))
    o=np.argsort(Qgt); a=Qgt[o]; T=1-Pgt[o]
    keep=np.concatenate([[True],np.diff(a)>1e-9]); a,T=a[keep],T[keep]
    idx=np.unique(np.linspace(0,len(a)-1,400).astype(int))
    return a[idx],np.clip(T[idx],0,1),adv

eps_plot=np.linspace(0,1.2*EPS0,260)
alpha_env=np.unique(np.concatenate([np.geomspace(1e-8,1e-2,120),np.linspace(1e-2,1-1e-5,300)]))
def f_ed(e,d,a): return np.maximum.reduce([np.zeros_like(a),1-d-math.exp(e)*a,math.exp(-e)*(1-d-a)])

fig,ax=plt.subplots(1,3,figsize=(11,3.1))
advs=[]
for p,c in zip(MECHS,BLUES):
    s=calib(p); o=mk(s,p); d=o.get_delta_for_epsilon(EPS0)
    prof=np.array([o.get_delta_for_epsilon(float(e)) for e in eps_plot])
    a,T,adv=roc(o); advs.append(adv)
    print(f'p={p:<6} I={STEPS} sigma={s:.5f} delta(10)={d:.3e} rel={abs(d-DELTA0)/DELTA0:.4f} adv={adv:.4f}')
    lbl=f'p={p}' + (' (Gaussian)' if p==1.0 else '')
    ax[0].semilogy(eps_plot,np.maximum(prof,1e-14),color=c,lw=1.4,label=lbl)
    ax[1].plot(a,T,color=c,lw=1.4)
ax[0].plot([EPS0],[DELTA0],'o',color=ORANGE,ms=5,label='contract')
ax[0].set_ylim(1e-13,3); ax[0].set_xlim(0,12); ax[0].set_xlabel(r'$\varepsilon$'); ax[0].set_ylabel(r'$\delta(\varepsilon)$')
ax[0].set_title('(a) privacy profiles'); ax[0].legend(fontsize=6,loc='lower left')
adv_env=(math.exp(EPS0)-1+2*DELTA0)/(math.exp(EPS0)+1)
ax[1].plot([0,1],[1,0],color='0.5',lw=0.7); ax[1].plot(alpha_env,f_ed(EPS0,DELTA0,alpha_env),':',color=ORANGE,lw=2)
ax[1].set_xlim(0,1); ax[1].set_ylim(0,1); ax[1].set_aspect('equal')
ax[1].set_xlabel(r'type-I error $\alpha$'); ax[1].set_ylabel(r'type-II error $T(\alpha)$'); ax[1].set_title('(b) trade-off curves')
xs=range(1,len(MECHS)+2); vals=advs+[adv_env]; cols=BLUES+[ORANGE]
ax[2].bar(list(xs),vals,color=cols,width=0.7)
for x,v in zip(xs,vals): ax[2].text(x,v+0.02,f'{v:.3f}',ha='center',va='bottom',fontsize=6,rotation=90)
ax[2].set_xticks(list(xs)); ax[2].set_xticklabels(['0.001','0.01','0.1','0.3','0.6','1',r'$A_{\varepsilon,\delta}$'],rotation=35,ha='right',fontsize=7)
ax[2].set_ylim(0,1.15); ax[2].set_ylabel('Adv'); ax[2].set_title('(c) advantage'); ax[2].set_xlabel('mechanism (p)')
fig.suptitle(f'Fig. 2 (inspection): advantage hierarchy at eps=10, delta=1e-12, I={STEPS} compositions',fontsize=9)
fig.tight_layout(rect=[0,0,1,0.95])
out='game_experiments/_cache_flpop/fig2_I10_inspect.png'
fig.savefig(out,dpi=150); print('wrote',out)
print('advantage spread:',min(advs),max(advs))
