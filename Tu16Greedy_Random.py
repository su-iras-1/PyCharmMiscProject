import csv
from collections import deque
from itertools import combinations
import random
import numpy as np
import torch
import torch.nn as nn

# ========================== 1. 全局配置与参数定义 ==========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SPOTS = 12  # 波位数 N = 12
NUM_BEAMS = 4  # 同时工作波束数 K = 4
SLOT_TIME = 0.01  # 跳波束时隙长度 10 ms
MAX_DELAY_TH = 400.0  # 排队时延阈值 400 ms
PACKET_SIZE_BIT = 10 * 1024  # 数据包大小 10 kbit
PACKET_SIZE_MBIT = PACKET_SIZE_BIT / 1e6
BANDWIDTH = 200e6  # 单波束带宽 200 MHz
TOTAL_POWER = 120.0  # 星上可用总功率 120 W
MAX_BEAM_POWER = 60.0  # 最大单波束功率 60 W
ORBIT_ALTITUDE = 570e3  # 轨道高度 570 km
FREQ = 20e9  # 载波频率 20 GHz
C = 3e8  # 光速
LAMBDA = C / FREQ
G_TX_DB = 40.0
G_RX_DB = 50.0
G_TX = 10 ** (G_TX_DB / 10.0)
G_RX = 10 ** (G_RX_DB / 10.0)
K_B = 1.38e-23
TEMP = 300.0
NOISE_POWER = K_B * TEMP * BANDWIDTH

ALL_ACTIONS = list(combinations(range(NUM_SPOTS), NUM_BEAMS))
ACTION_DIM = len(ALL_ACTIONS)

spot_positions = np.array(
    [
        [-300e3, 300e3],
        [-100e3, 300e3],
        [100e3, 300e3],
        [300e3, 300e3],
        [-300e3, 0.0],
        [-100e3, 0.0],
        [100e3, 0.0],
        [300e3, 0.0],
        [-300e3, -300e3],
        [-100e3, -300e3],
        [100e3, -300e3],
        [300e3, -300e3],
    ]
)
sat_pos = np.array([0.0, 0.0, ORBIT_ALTITUDE])

SPOT_BASE_TRAFFIC_MBPS = np.array(
    [300, 500, 800, 1100, 400, 900, 1200, 600, 200, 450, 750, 350], dtype=float
)


def time_factor_hour(hour):
    if 9.0 <= hour < 11.0:
        return 0.7 + 0.3 * (hour - 9.0) / 2.0
    elif 11.0 <= hour < 12.0:
        return 1.0 - 0.1 * (hour - 11.0)
    elif 12.0 <= hour <= 14.0:
        return 0.9 - 0.2 * (hour - 12.0) / 2.0
    else:
        return 0.7


# ========================== 2. 信道模型计算 ==========================
def antenna_gain_val(theta, theta_3dB=0.035):
    try:
        from numpy import jv
    except ImportError:
        from scipy.special import jv
    if abs(theta) < 1e-6:
        return 1.0
    u = 2.07123 * np.sin(theta) / np.sin(theta_3dB)
    J1 = jv(1, u)
    J3 = jv(3, u)
    G = (J1 / (2.0 * u) + 36.0 * J3 / (u**3)) ** 2
    return max(G, 1e-6)


INTERFERENCE_GAIN_MATRIX = np.zeros((NUM_SPOTS, NUM_SPOTS))
for i in range(NUM_SPOTS):
    pos_i = spot_positions[i]
    d_i = np.linalg.norm(pos_i - sat_pos[:2])
    dist_i = np.sqrt(d_i**2 + sat_pos[2] ** 2)
    for j in range(NUM_SPOTS):
        if i == j:
            continue
        pos_j = spot_positions[j]
        d_j = np.linalg.norm(pos_j - sat_pos[:2])
        dist_j = np.sqrt(d_j**2 + sat_pos[2] ** 2)
        d_ij = np.linalg.norm(pos_i - pos_j)
        if d_ij < 1e-3:
            continue
        cos_theta = (d_i**2 + d_j**2 + 2 * sat_pos[2] ** 2 - d_ij**2) / (
            2 * dist_i * dist_j
        )
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        G_theta = antenna_gain_val(theta)
        INTERFERENCE_GAIN_MATRIX[i, j] = (
            G_TX * G_theta * (LAMBDA**2)
        ) / ((4 * np.pi * d_ij) ** 2)

PATH_LOSS = np.zeros(NUM_SPOTS)
for i in range(NUM_SPOTS):
    d_i = np.linalg.norm(spot_positions[i] - sat_pos[:2])
    dist_i = np.sqrt(d_i**2 + sat_pos[2] ** 2)
    PATH_LOSS[i] = (4 * np.pi * dist_i / LAMBDA) ** 2


def compute_capacity_fast(active_indices, power_allocation):
    capacities = np.zeros(NUM_SPOTS)
    P_vec = np.zeros(NUM_SPOTS)
    for i in active_indices:
        P_vec[i] = power_allocation[i]

    interference = np.dot(P_vec, INTERFERENCE_GAIN_MATRIX)

    for i in active_indices:
        signal = G_TX * power_allocation[i] * G_RX / PATH_LOSS[i]
        sinr = signal / (interference[i] + NOISE_POWER + 1e-12)
        capacities[i] = BANDWIDTH * np.log2(1.0 + sinr) / 1e6
    return capacities


# ========================== 3. 仿真环境定义 ==========================
class BeamHoppingEnv:

    def __init__(self, history_len=40):
        self.history_len = history_len
        self.reset()

    def reset(self):
        self.real_queues = [deque() for _ in range(NUM_SPOTS)]
        self.nonreal_queue = np.zeros(NUM_SPOTS, dtype=float)
        self.total_real_arrived = np.zeros(NUM_SPOTS, dtype=float)
        self.total_nonreal_arrived = np.zeros(NUM_SPOTS, dtype=float)
        self.served_real = np.zeros(NUM_SPOTS, dtype=float)
        self.served_nonreal = np.zeros(NUM_SPOTS, dtype=float)

        self.slot_index = 0
        self.history = []

        self.state_history_real = np.zeros((NUM_SPOTS, self.history_len))
        self.state_history_nonreal = np.zeros((NUM_SPOTS, self.history_len))
        self.satisfaction_vec = np.ones(NUM_SPOTS, dtype=np.float32)

        return self._get_state()

    def _get_state(self):
        real_len = np.array(
            [len(q) for q in self.real_queues], dtype=np.float32
        )
        nonreal_len = self.nonreal_queue.astype(np.float32)

        self.state_history_real = np.roll(self.state_history_real, -1, axis=1)
        self.state_history_nonreal = np.roll(
            self.state_history_nonreal, -1, axis=1
        )
        self.state_history_real[:, -1] = real_len
        self.state_history_nonreal[:, -1] = nonreal_len

        state_matrix = np.stack(
            [self.state_history_real, self.state_history_nonreal], axis=0
        ).astype(np.float32)
        return state_matrix, self.satisfaction_vec.copy()

    def step(self, action_vec, current_hour=11.0):
        self.slot_index += 1
        tf = time_factor_hour(current_hour)

        arrival_mbps = SPOT_BASE_TRAFFIC_MBPS * tf
        packets_expected = (arrival_mbps * SLOT_TIME) / PACKET_SIZE_MBIT

        real_arrive = np.random.poisson(packets_expected * 0.5)
        nonreal_arrive = np.random.poisson(packets_expected * 0.5)

        self.total_real_arrived += real_arrive
        self.total_nonreal_arrived += nonreal_arrive
        self.nonreal_queue += nonreal_arrive

        for spot_idx, packets in enumerate(real_arrive):
            for _ in range(int(packets)):
                self.real_queues[spot_idx].append(self.slot_index)

        active = np.where(action_vec == 1)[0]
        if len(active) == 0:
            active = np.random.choice(NUM_SPOTS, NUM_BEAMS, replace=False)

        weights = {}
        for i in active:
            real_cnt = len(self.real_queues[i])
            nonreal_cnt = self.nonreal_queue[i]
            total_packets = real_cnt + nonreal_cnt
            if real_cnt > 0:
                avg_d = np.mean(
                    [
                        (self.slot_index - p) * SLOT_TIME * 1000.0
                        for p in self.real_queues[i]
                    ]
                )
            else:
                avg_d = 1.0
            weights[i] = total_packets * max(avg_d, 1.0)

        total_weight = sum(weights.values())
        power_allocation = {}
        if total_weight == 0:
            for i in active:
                power_allocation[i] = TOTAL_POWER / len(active)
        else:
            for i in active:
                power_allocation[i] = min(
                    (weights[i] / total_weight) * TOTAL_POWER, MAX_BEAM_POWER
                )

        capacities = compute_capacity_fast(active, power_allocation)

        served_real_slot = np.zeros(NUM_SPOTS)
        served_nonreal_slot = np.zeros(NUM_SPOTS)

        for i in active:
            cap_mbps = capacities[i]
            max_packets_served = int((cap_mbps * SLOT_TIME) / PACKET_SIZE_MBIT)

            q = self.real_queues[i]
            real_serve_cnt = min(len(q), max_packets_served)
            for _ in range(real_serve_cnt):
                q.popleft()
            served_real_slot[i] = real_serve_cnt
            rem_capacity_packets = max_packets_served - real_serve_cnt

            if rem_capacity_packets > 0 and self.nonreal_queue[i] > 0:
                nonreal_serve_cnt = min(
                    self.nonreal_queue[i], rem_capacity_packets
                )
                self.nonreal_queue[i] -= nonreal_serve_cnt
                served_nonreal_slot[i] = nonreal_serve_cnt

        self.served_real += served_real_slot
        self.served_nonreal += served_nonreal_slot

        dropped_packets = 0
        for i in range(NUM_SPOTS):
            q = self.real_queues[i]
            while q:
                if (
                    self.slot_index - q[0]
                ) * SLOT_TIME * 1000.0 > MAX_DELAY_TH:
                    q.popleft()
                    dropped_packets += 1
                else:
                    break

        all_delays = []
        for i in range(NUM_SPOTS):
            for arr in self.real_queues[i]:
                all_delays.append(
                    (self.slot_index - arr) * SLOT_TIME * 1000.0
                )
        avg_delay = np.mean(all_delays) if all_delays else 5.0

        throughput_mbps = (
            served_nonreal_slot.sum() * PACKET_SIZE_MBIT
        ) / SLOT_TIME

        tot_arrived = self.total_real_arrived + self.total_nonreal_arrived
        tot_served = self.served_real + self.served_nonreal
        self.satisfaction_vec = np.where(
            tot_arrived > 0, tot_served / (tot_arrived + 1e-8), 1.0
        )
        self.satisfaction_vec = np.clip(self.satisfaction_vec, 0.0, 1.0)
        avg_satisfaction = np.mean(self.satisfaction_vec)

        self.history.append(
            {
                "delay": avg_delay,
                "throughput_mbps": throughput_mbps,
                "satisfaction": avg_satisfaction,
            }
        )

        next_state = self._get_state()
        return next_state, (0, 0, 0), False, {}


# ========================== 4. 网络模型与 Agent 定义 ==========================
class DQN_CNN(nn.Module):

    def __init__(self, input_channels=2, action_dim=ACTION_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 8, kernel_size=(1, 20), padding=0
        )
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 5), padding=0)
        self.fc1 = nn.Linear(16 * 12 * 17, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DQN_FC(nn.Module):

    def __init__(self, input_dim=NUM_SPOTS, action_dim=ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MultiObjectiveMoEDQN:

    def __init__(self):
        self.q_net1 = DQN_CNN().to(DEVICE)
        self.q_net2 = DQN_CNN().to(DEVICE)
        self.q_net3 = DQN_FC().to(DEVICE)

    def load_model(self, path_prefix="moe_dqn_final"):
        self.q_net1.load_state_dict(
            torch.load(f"{path_prefix}_q1.pth", map_location=DEVICE)
        )
        self.q_net2.load_state_dict(
            torch.load(f"{path_prefix}_q2.pth", map_location=DEVICE)
        )
        self.q_net3.load_state_dict(
            torch.load(f"{path_prefix}_q3.pth", map_location=DEVICE)
        )
        print(f"--> 成功加载训练权重 {path_prefix}_*.pth！")

    def act(self, state, eval_mode=True):
        state_matrix, state_sat = state
        sm_t = torch.FloatTensor(state_matrix).unsqueeze(0).to(DEVICE)
        ss_t = torch.FloatTensor(state_sat).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q1 = self.q_net1(sm_t).squeeze(0)
            q2 = self.q_net2(sm_t).squeeze(0)
            q3 = self.q_net3(ss_t).squeeze(0)

            q1_norm = (q1 - q1.mean()) / (q1.std() + 1e-6)
            q2_norm = (q2 - q2.mean()) / (q2.std() + 1e-6)
            q3_norm = (q3 - q3.mean()) / (q3.std() + 1e-6)

            q_total = 0.4 * q1_norm + 0.3 * q2_norm + 0.3 * q3_norm
            idx = torch.argmax(q_total).item()

        action_vec = np.zeros(NUM_SPOTS, dtype=int)
        action_vec[list(ALL_ACTIONS[idx])] = 1
        return action_vec, idx


# ========================== 5. 对比策略定义 ==========================
def act_greedy(env):
    queue_lens = [
        len(env.real_queues[i]) + env.nonreal_queue[i] for i in range(NUM_SPOTS)
    ]
    top4_spots = np.argsort(queue_lens)[-NUM_BEAMS:]
    action_vec = np.zeros(NUM_SPOTS, dtype=int)
    action_vec[top4_spots] = 1
    return action_vec


def act_random():
    idx = np.random.randint(ACTION_DIM)
    action_vec = np.zeros(NUM_SPOTS, dtype=int)
    action_vec[list(ALL_ACTIONS[idx])] = 1
    return action_vec


# ========================== 6. 主导出函数 ==========================
def generate_fig16_full_comparison_data(
    model_prefix="moe_dqn_final", num_eval_slots=500
):
    print(
        f"=== 开始提取整点数据并导出 figure16_full_comparison.csv ==="
    )

    moe_agent = MultiObjectiveMoEDQN()
    try:
        moe_agent.load_model(model_prefix)
    except FileNotFoundError:
        print(
            f"[错误] 未找到 {model_prefix}_*.pth 权重文件！请确保该文件在当前文件夹下。"
        )
        return

    env_moe = BeamHoppingEnv()
    env_greedy = BeamHoppingEnv()
    env_random = BeamHoppingEnv()

    np.random.seed(2026)
    random.seed(2026)

    s_moe = env_moe.reset()
    s_greedy = env_greedy.reset()
    s_rand = env_random.reset()

    hours_targets = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    target_slots = {}
    for h in hours_targets:
        slot_idx = int(round((h - 9.0) / 5.0 * (num_eval_slots - 1)))
        target_slots[slot_idx] = f"{int(h)}:00"

    hourly_records = []

    for t in range(num_eval_slots):
        current_hour = 9.0 + (t / float(num_eval_slots - 1)) * 5.0

        a_moe, _ = moe_agent.act(s_moe, eval_mode=True)
        a_greedy = act_greedy(env_greedy)
        a_rand = act_random()

        s_moe, _, _, _ = env_moe.step(a_moe, current_hour=current_hour)
        s_greedy, _, _, _ = env_greedy.step(a_greedy, current_hour=current_hour)
        s_rand, _, _, _ = env_rand.step(a_rand, current_hour=current_hour)

        if t in target_slots:
            time_str = target_slots[t]

            # 记录 MA-DRL (MoE)
            hourly_records.append(
                {
                    "Time": time_str,
                    "Algorithm": "MA-DRL (MoE)",
                    "Avg_Delay_ms": round(env_moe.history[-1]["delay"], 2),
                    "Avg_Throughput_Mbps": round(
                        env_moe.history[-1]["throughput_mbps"], 2
                    ),
                    "Satisfaction": round(
                        env_moe.history[-1]["satisfaction"], 4
                    ),
                }
            )
            # 记录 Greedy
            hourly_records.append(
                {
                    "Time": time_str,
                    "Algorithm": "Greedy (Max-Queue)",
                    "Avg_Delay_ms": round(env_greedy.history[-1]["delay"], 2),
                    "Avg_Throughput_Mbps": round(
                        env_greedy.history[-1]["throughput_mbps"], 2
                    ),
                    "Satisfaction": round(
                        env_greedy.history[-1]["satisfaction"], 4
                    ),
                }
            )
            # 记录 Random
            hourly_records.append(
                {
                    "Time": time_str,
                    "Algorithm": "Random",
                    "Avg_Delay_ms": round(env_rand.history[-1]["delay"], 2),
                    "Avg_Throughput_Mbps": round(
                        env_rand.history[-1]["throughput_mbps"], 2
                    ),
                    "Satisfaction": round(
                        env_rand.history[-1]["satisfaction"], 4
                    ),
                }
            )

    csv_filename = "figure16_full_comparison.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Time",
                "Algorithm",
                "Avg_Delay_ms",
                "Avg_Throughput_Mbps",
                "Satisfaction",
            ],
        )
        writer.writeheader()
        writer.writerows(hourly_records)

    print(
        f"--> [成功] 数据已保存至: {csv_filename} (包含 3 种算法共 {len(hourly_records)} 行对比数据)"
    )


if __name__ == "__Tu16Greedy_Random__":
    generate_fig16_full_comparison_data(
        model_prefix="moe_dqn_final", num_eval_slots=500
    )