import numpy as np
from network.base_station import BaseStation as BS

# --------- ① 先取得 obs key 长度，用于裁剪 -----------------
_OBS_KEYS = BS.get_all_obs_keys() if hasattr(BS, "get_all_obs_keys") else None
_OBS_LEN  = len(_OBS_KEYS) if _OBS_KEYS is not None else None
# ----------------------------------------------------------

class SimplePolicy:
    pre_sm2_steps = 10
    pre_sm3_steps = 50

    def __init__(self, action_space, num_agents):
        self.act_space  = action_space
        self.num_agents = num_agents
        self._sleep_steps = [0] * num_agents
        self._exp_len = None            # <-- 缓存 annotate_obs 真正需要的长度

        # 天线动作的“保持不变”索引
        self._no_ant_switch = (BS.num_ant_switch_opts - 1) // 2

    def _parse_obs(self, raw: np.ndarray):
        """
        调 BS.annotate_obs()，如果长度对不上就不断裁剪直到成功。
        把成功时的长度缓存，以后直接用。
        """
        # 已知长度 -> 直接裁剪
        if self._exp_len is not None:
            return BS.annotate_obs(raw[: self._exp_len])

        # 不知道长度 -> 由长到短试
        for cut in range(len(raw), 0, -1):
            try:
                info = BS.annotate_obs(raw[:cut])
            except AssertionError:
                continue            # 长度还不对，继续减 1
            else:
                self._exp_len = cut  # 缓存
                return info

        # 全试完还不行 -> 真的有问题
        raise ValueError("Cannot annotate observation: len mismatch")

    # ---------------- 工具 ---------------- #
    @staticmethod
    def _safe(info: dict, key: str, default=0.0):
        return info[key] if key in info else default

    # -------------- 单基站决策 ------------- #
    def _single_act(self, idx: int, raw_obs: np.ndarray):

        info = self._parse_obs(raw_obs)    

        sm        = int(info['sleep_mode'])
        next_sm   = int(info['next_sleep_mode'])
        wake_time = self._safe(info, 'wakeup_time')

        thrp_serv  = self._safe(info, 'serving_sum_rate')
        thrp_req_s = self._safe(info, 'serving_sum_rate_req')
        thrp_req_i = self._safe(info, 'idle_sum_rate_req')

        new_sm     = sm
        ant_switch = self._no_ant_switch
        conn_mode  = 0 if sm else 2

        if sm:                                    # 睡眠状态
            self._sleep_steps[idx] += 1
            if thrp_req_i:                        # 有空闲 UE 需要服务
                new_sm = 0
                if wake_time < 0.005:
                    conn_mode = 2
            elif sm == 1 and self._sleep_steps[idx] >= self.pre_sm2_steps:
                new_sm = 2
            elif sm == 2 and self._sleep_steps[idx] >= self.pre_sm3_steps:
                new_sm = 3
        else:                                     # 激活状态
            self._sleep_steps[idx] = 0
            if not thrp_req_s:
                new_sm = 1
            else:
                ratio = thrp_serv / max(thrp_req_s, 1e-6)
                if ratio > 2:
                    ant_switch = max(self._no_ant_switch - 1, 0)
                elif ratio < 1:
                    ant_switch = min(self._no_ant_switch + 1,
                                     BS.num_ant_switch_opts - 1)

        # gym.MultiDiscrete 的顺序：[ant_switch, sleep_mode, conn_mode]
        return [ant_switch, new_sm, conn_mode]

    # -------------- 批量决策 --------------- #
    def act(self, obs_batch, **_):
        return [self._single_act(i, ob) for i, ob in enumerate(obs_batch)]


# --------- 下面三个派生策略仅修改等待步数 -------------------- #
class SimplePolicySM1Only(SimplePolicy):
    pre_sm2_steps = int(1e9)
    pre_sm3_steps = int(1e9)

class SimplePolicyNoSM3(SimplePolicy):
    pre_sm3_steps = int(1e9)

class SleepyPolicy(SimplePolicy):
    pre_sm2_steps = 2
    pre_sm3_steps = 6
