"""Geometry-cached vector power updates for the canonical orthogonal V1 model.

This is an algebraic evaluation backend, not a different channel model.
Geometry/DD arrays already exist in BaseGains; counters track queried entries,
not actual waveform executions or end-to-end preprocessing work.
"""
from collections import OrderedDict
from dataclasses import asdict,replace
import numpy as np
from .model import LinkTables,noise_power,denominator_guard,bandwidth,radar_hardware_gain
from .fbl import chi_from_gamma
from .llr import soft_mean,soft_var0


def _signature(cfg):
    radio=asdict(cfg.radio);radio.pop('rho');radio.pop('rho_by_uav')
    return (radio,asdict(cfg.waveform),asdict(cfg.interference),asdict(cfg.comm),
            asdict(cfg.detect),asdict(cfg.dd),asdict(cfg.waveform_impairments),asdict(cfg.scale))


class PowerTableCache:
    def __init__(self,cfg,base,max_cached=128):
        if (cfg.interference.coupling!='shared_spectrum' or cfg.comm.interference_model!='orthogonal'
            or cfg.comm.mac_model!='serial' or cfg.radio.isac_power_model!='sensing_only'
            or cfg.interference.sense_gate_by_active_tx or cfg.waveform_impairments.enable
            or cfg.detect.soft_stat_model!='llr'):
            raise ValueError('cached power tables require canonical independent-waveform orthogonal LLR physics')
        if max_cached<1:raise ValueError('cache capacity must be positive')
        self.base=base;self.signature=_signature(cfg);self.max_cached=max_cached
        self.cache=OrderedDict();self.table_builds=0;self.cache_hits=0;self.fine_updates=0
        self.refined_seen=np.zeros_like(base.target_gain,dtype=bool)
        M=cfg.scale.M
        self.offdiag=~np.eye(M,dtype=bool)
        self.valid=np.broadcast_to(self.offdiag[:,:,None],base.target_gain.shape).copy()
        if cfg.dd.use_otfs_bin_validity:self.valid &= base.valid_dd
        proc=cfg.waveform.N*cfg.waveform.L if cfg.detect.sensing_processing_gain is None else cfg.detect.sensing_processing_gain
        self.unit_signal=base.target_gain*proc*radar_hardware_gain(cfg)
        self.collision=1./np.maximum(base.dd_collision_count,1.)**cfg.dd.dd_collision_alpha

    def __call__(self,cfg,base,dd_gain=None):
        if base is not self.base or _signature(cfg)!=self.signature:
            raise ValueError('power cache belongs to one geometry and fixed physical configuration')
        M=cfg.scale.M;r=cfg.radio;c=cfg.comm;d=cfg.detect
        rho=np.asarray(r.rho_by_uav if r.rho_by_uav is not None else [r.rho]*M,dtype=float)
        if rho.shape!=(M,) or not np.all(np.isfinite(rho)) or np.any((rho<=0)|(rho>=1)):
            raise ValueError('invalid power fractions')
        key=tuple(rho)
        if key in self.cache:
            coarse,unweighted=self.cache.pop(key);self.cache[key]=(coarse,unweighted);self.cache_hits+=1
        else:
            P=np.full(M,r.P_default);ps=rho*P;pc=(1-rho)*P
            n0=noise_power(cfg);eps=denominator_guard(cfg,n0);B=bandwidth(cfg)
            field=ps@base.direct_gain
            own=ps[:,None]*base.direct_gain
            denominator=n0+c.comm_leakage_from_sensing*(field[None,:]-own)+c.comm_direct_leakage_factor*own+eps
            active=base.edge_mask & self.offdiag
            gamma_comm=np.where(active,pc[:,None]*base.direct_gain/denominator,0.)
            rate=B*np.log2(1+gamma_comm)
            chi=np.where(active,chi_from_gamma(cfg,gamma_comm,max(2**(c.R_min/B)-1,1e-12)),0.)
            chi=np.clip(chi,0.,1.)
            feasible=active & (rate.T>=c.R_min)
            if c.enforce_chi_min:feasible &= chi.T>=c.chi_min
            residual=np.broadcast_to(r.residual_self_factor*P[None,:]+
                10**(-cfg.interference.direct_cancellation_db/10)*field[None,:],(M,M)).copy()
            rinr=residual/(n0+eps);rinr[~self.offdiag]=0.
            sigma=d.soft_sigma0*np.sqrt(1+r.rinr_sigma_factor*rinr)
            raw=ps[:,None,None]*self.unit_signal
            unweighted=np.where(self.valid,raw/(n0+residual[:,:,None]+eps)*self.collision,0.)
            gamma=unweighted*base.dd_frac_loss if cfg.dd.enable_dd_fractional_penalty else np.where(self.valid,raw/(n0+residual[:,:,None]+eps)*self.collision,0.)
            coarse=LinkTables(gamma_comm,rate,chi,feasible,np.where(self.valid,raw/(n0+eps),0.),
                gamma,rinr,np.asarray(soft_mean(cfg,gamma)),sigma,np.asarray(soft_var0(cfg,gamma,sigma[:,:,None]**2)))
            self.cache[key]=(coarse,unweighted);self.table_builds+=1
            if len(self.cache)>self.max_cached:self.cache.popitem(last=False)
        if dd_gain is None or not cfg.dd.enable_dd_fractional_penalty:return coarse
        gain=np.asarray(dd_gain)
        if gain.shape!=base.dd_frac_loss.shape:raise ValueError('invalid DD gain shape')
        mask=(gain!=base.dd_frac_loss)&self.valid
        self.refined_seen |= mask;self.fine_updates+=int(mask.sum())
        if not mask.any():return coarse
        gamma=coarse.gamma_sense.copy();mu=coarse.mu_soft.copy();var=coarse.var0_q.copy()
        gamma[mask]=unweighted[mask]*gain[mask]
        mu[mask]=soft_mean(cfg,gamma[mask]);var[mask]=soft_var0(cfg,gamma[mask],1.)
        return replace(coarse,gamma_sense=gamma,mu_soft=mu,var0_q=var)

    def refine_pool(self,cfg,pool):
        gain=self.base.dd_frac_loss.copy()
        for q,links in pool.items():
            for i,j in links:gain[i,j,q]=self.base.eta_fine[i,j,q]
        return self(cfg,self.base,dd_gain=gain)
