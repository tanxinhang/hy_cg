"""直连干扰口径：耦合方式、对消深度常数、门控开关。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Interference:
    """通信与感知如何共用同一段频谱。

    ISAC 网络里一架 UAV 把**单一**联合波形辐射进同一个频段，所以节点 ``j``
    上的接收机对同一批并发发射机要看见两次：一次对着它的通信信号，一次对着
    它的感知回波。因此"给通信完整的干扰项、却让感知几乎观察不到干扰"的模型
    根本不是 ISAC 模型 —— 它恰好抹掉了让 ISAC 变难的耦合，也抹掉了两个功能
    需要联合设计的全部理由。

    ``coupling`` 决定怎么记账。

    * ``"legacy"`` —— 通信干扰跟随 ``comm.interference_model``，而感知干扰是
      一组**解耦**的残余地板（``radio.residual_*``）。这些地板既不跟随当前
      发射集、也不复用直连增益，而且其中最强的干扰源 —— 照射机自己的直连
      路径 —— 被明确排除在求和之外。只为与冻结结果逐位可比而保留。

    * ``"shared_spectrum"`` —— 两个接收机都由**同一个**干扰场构成，发射集与
      直连增益也相同。在节点 ``j`` 上，

          I_comm(i,j) = sum_{k != i, j} ( P_comm[k] + c_leak * P_sense[k] ) g_kj
          I_sense(j)  = kappa_self * P[j] + kappa_dc * sum_{k != j} P_tx[k] g_kj

      其中 ``P_tx[k] = P_sense[k] + P_comm[k]`` 是 UAV ``k`` 辐射出来的 ISAC
      功率。这个耦合有两条性质值得明说。

      1. 照射机 ``i`` 在感知侧**就是**干扰源：它的直连路径与它自己照出的目标
         回波竞争，这正是双站感知经典的近远问题。而在通信侧，同一架 UAV 是
         想要的信源。这是**角色**的差别，不是物理的差别。
      2. ``kappa_dc`` 是感知接收机能达到的直连对消能力，**它不再由本段声明**。

    关于第 2 点，这里必须把话说完整，因为这个字段曾经存在过又退役了。原段曾有
    一个常数 ``direct_cancellation_db``（出厂 40 dB），断言接收机从聚合直连场里
    消掉固定的 40 dB。它被删了，理由不是"40 dB 太小"，而是**生产链路里没有
    任何接收机实现支撑它**：那个数字是从需求反推出来的（推导与文献锚点见
    ``tools/derive_kappa_pilot_budget.py``，退役记录见 ``docs/DEV_STANDARD.md``
    §5.3），不是一个被测量出来的量。而它恰好撑着 SINR 的
    分母 —— 把它取成 0 dB 会让 P_D 掉到 P_FA，也就是完全失去检测能力 ——
    所以挂在它上面的任何"增益"都是分母上的记账，不是系统性能。

    删除之后，直连对消**只剩一个来源**：接收机的实测残余，由
    :mod:`isac_sim.receiver.cancellation`（TP-UIC）产出，并以
    ``residual_fraction_by_receiver`` / ``residual_power_by_receiver_target``
    喂进链路表。没喂进来就是**没有对消**（直连场全额存活），不是"默认 40 dB"。
    这是刻意的：宁可让数字难看，也不让一个没有实现支撑的常数替系统拿分。
    残余的解析口径与优先级见
    :mod:`isac_sim.sensing.model.link_tables.residual`。


    感知波形按假设是**持续辐射**的（这就是 ISAC 的前提），所以除非打开
    ``sense_gate_by_active_tx``，感知干扰与上报调度无关；被"谁真的在报"门控的
    只有通信载荷。
    """

    coupling: str = "shared_spectrum"
    # 感知侧也跟随当前发射集。物理上没必要（照射是持续的），但用来隔离效应很有用。
    sense_gate_by_active_tx: bool = False
