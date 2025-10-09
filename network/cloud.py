# network/cloud.py
import numpy as np
from collections import defaultdict
from . import config
from utils import *
from config import *

class CloudPowerModel:
    """Centralised GPP """
    def __init__(self, net):
        self.net = net                       # MultiCellNetwork 
        self.reset()

    # -------------------------------------------------------
    def reset(self):
        self._time      = 0.0
        self.energy_J   = 0.0                # cloud energy (J)
        self.P_fixed    = config.P_cloud_fixed
        self.P_tr       = 0.0                # fronthaul transmit
        self.P_proc     = 0.0                # GPP processing
        self._stats_rec = defaultdict(float) 

    # -------------------------------------------------------
    def _compute_gops(self):
        """
        GOPS :  C_GPP = Σ_l ( Cmod + Ccoding + Cnetwork )
        """
        BW_ref = 20e6
        SE_ref = 6.0
        se_avg = 3.3
        W_r    = self.net.bss[0].bandwidth / BW_ref  

        # Σ_l … Σ_AP
        Cmod_sum = 0.0
        Ccod_sum = 0.0
        Cnet_sum = 0.0
        p_fh_sum = 0.0                             # fronthaul 

        for bs in self.net.bss.values():
            UE_cnt = len(bs.ues)
            if UE_cnt:
                # se_avg = np.mean([ue.SE for ue in bs.ues.values()])
                SE_r   = se_avg / SE_ref
            else:
                SE_r = 0.0

            Cmod_sum += 1.3 * W_r * bs.num_ant
            Ccod_sum += 5.2 * W_r * SE_r * UE_cnt
            Cnet_sum += 8.0 * W_r * SE_r

        C_GPP = Cmod_sum
        return C_GPP

    # -------------------------------------------------------
    def step_power(self, dt):
        
    
        C_GPP = self._compute_gops()

        # P_cloud  (8)
        P_proc = ( config.P_cloud_proc0 +
                   config.proc_slope_GPP * C_GPP / config.C_GPP_max )

        P_cool = (1.0 / config.sigmaCool) * ( config.P_olt +
                                              config.proc_slope_GPP +
                                              config.P_cloud_proc0 )

        P_total = self.P_fixed + (1/config.sigmaCool) * P_proc

        
        self.P_proc = P_proc
        
        self.energy_J += P_total * dt
        self._time    += dt

        
        if EVAL:
            self._stats_rec['time']       = self._time
            self._stats_rec['P_fixed']    += self.P_fixed * dt
            self._stats_rec['P_proc']     += P_proc * dt
            self._stats_rec['energy_J']   = self.energy_J

    # -------------------------------------------------------
    def avg_power(self):
        return self.energy_J / max(self._time, 1e-9)
