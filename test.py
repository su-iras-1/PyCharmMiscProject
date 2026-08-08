import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from itertools import combinations

# ========================== 固定随机种子 ==========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== 系统参数（表1 & 表2） ==========================
NUM_SPOTS = 12             # 波位数
NUM_BEAMS = 4              # 波束数
SLOT_TIME = 0.01           # 10 ms
MAX_DELAY = 400            # ms
PACKET_SIZE = 10 * 1024    # 10 kbit -> bit
BANDWIDTH = 200e6          # 200 MHz
TOTAL_POWER = 120.0        # 120 W
MAX_BEAM_POWER = 60.0      # 60 W
ORBIT_ALTITUDE = 570e3     # 570 km
FREQ = 20e9                # 20 GHz
C = 3e8
LAMBDA = C / FREQ
G_TX = 10 ** (40 / 10)     # 40 dB -> 线性增益 10000
G_RX = 10 ** (50 / 10)     # 50 dB -> 线性增益 100000
K_B = 1.38e-23
TEMP = 300
NOISE_POWER = K_B * TEMP * BANDWIDTH # 噪声功率 (W)

# 生成固定波位位置
COVERAGE_RADIUS = 500e3
spot_positions = np.random.uniform(-COVERAGE_RADIUS, COVERAGE_RADIUS, (NUM_SPOTS, 2))
sat_pos = np.array([0, 0, ORBIT_ALTITUDE])

# 动作空间：C(12, 4) = 495
ALL_ACTIONS = list(combinations(range(NUM_SPOTS), NUM_BEAMS))
ACTION_DIM = len(ALL_ACTIONS)

# 业务基准参数
spot_means = np.random.uniform(100, 250, NUM_SPOTS)

def time_factor_hour(hour):
    if 8 <= hour < 10:
        return 0.5 + 0.5 * (hour - 8) / 2
    elif 10 <= hour < 12:
        return 1.0 + 0.2 * (hour - 10)
    elif 12 <= hour < 14:
        return 1.2 - 0.2 * (hour - 12)
    elif 14 <= hour < 18:
        return 1.0
    else:
        return 0.4

# 预计算天线增益与干扰矩阵
def  antenna_gain_val(theta, theta_3dB=0.2):
    try:
        from numpy import jv
    except ImportError:
        from scipy.special import jv
    if theta == 0:
        return 1.0
    u = 2.07123 * np.sin(theta) / np.sin(theta_3dB)
    J1 = jv(1, u)
    J3 = jv(3, u)
    G = (J1 / (2 * u) + 36 * J3 / (u ** 3)) ** 2
    return G * G_TX

INTERFERENCE_GAIN_MATRIX = np.zeros((NUM_SPOTS, NUM_SPOTS))
for i in range(NUM_SPOTS):
    pos_i = spot_positions[i]
    d_i = np.linalg.norm(pos_i - sat_pos[:2])
    dist_i = np.sqrt(d_i ** 2 + sat_pos[2] ** 2)
    for j in range(NUM_SPOTS):
        if i == j:
            continue
        pos_j = spot_positions[j]
        d_j = np.linalg.norm(pos_j - sat_pos[:2])
        dist_j = np.sqrt(d_j ** 2 + sat_pos[2] ** 2)
        d_ij = np.linalg.norm(pos_i - pos_j)
        if d_ij == 0:
            continue
        cos_theta = (d_i ** 2 + d_j ** 2 + 2 * sat_pos[2] ** 2 - d_ij ** 2) / (2 * dist_i * dist_j)
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        G_theta = antenna_gain_val(theta)
        INTERFERENCE_GAIN_MATRIX[i, j] = (G_TX * G_theta * LAMBDA ** 2) / ((4 * np.pi * d_ij) ** 2)

PATH_LOSS = np.zeros(NUM_SPOTS)
for i in range(NUM_SPOTS):
    d_i = np.linalg.norm(spot_positions[i] - sat_pos[:2])
    dist_i = np.sqrt(d_i ** 2 + sat_pos[2] ** 2)
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
        capacities[i] = BANDWIDTH * np.log2(1 + sinr)  # bps
    return capacities

# ========================== 低轨卫星跳波束环境类 ==========================
class BeamHoppingEnv:
    def __init__(self, history_len=40):
        self.history_len = history_len
        self.reset()

    def reset(self):
        self.real_queues = [deque() for _ in range(NUM_SPOTS)]      # 存储实时包的到达时隙
        self.nonreal_queues = np.zeros(NUM_SPOTS, dtype=int)        # 存储非实时包个数

        # 统计变量（全部统一为数据包数量）
        self.total_arrived = np.zeros(NUM_SPOTS, dtype=float)
        self.total_served = np.zeros(NUM_SPOTS, dtype=float)
        self.total_served_nonreal_packets = 0.0

        self.slot_index = 0
        self.history = []

        # 状态矩阵初始化
        self.state_real_history = np.zeros((NUM_SPOTS, self.history_len), dtype=np.float32)
        self.state_nonreal_history = np.zeros((NUM_SPOTS, self.history_len), dtype=np.float32)
        self.satisfaction_vec = np.ones(NUM_SPOTS, dtype=np.float32) # 初始满意度 1.0

        return self._get_state()

    def _get_state(self):
        real_len = np.array([len(q) for q in self.real_queues], dtype=np.float32)
        nonreal_len = self.nonreal_queues.astype(np.float32)

        self.state_real_history = np.roll(self.state_real_history, -1, axis=1)
        self.state_nonreal_history = np.roll(self.state_nonreal_history, -1, axis=1)
        self.state_real_history[:, -1] = real_len
        self.state_nonreal_history[:, -1] = nonreal_len

        # 卷积输入: (2, 12, 40)
        state_conv = np.stack([self.state_real_history, self.state_nonreal_history], axis=0)

        # 满意度向量公式 (24): 已服务包数 / 到达包数
        for i in range(NUM_SPOTS):
            if self.total_arrived[i] > 0:
                self.satisfaction_vec[i] = self.total_served[i] / self.total_arrived[i]
            else:
                self.satisfaction_vec[i] = 1.0

        return state_conv, self.satisfaction_vec.copy()

    def step(self, action_vec):
        self.slot_index += 1
        hour = 9.0 + (self.slot_index * SLOT_TIME) / 3600.0
        tf = time_factor_hour(hour)

        # 1. 业务到达（按包计算）
        real_arrive = np.random.poisson(spot_means * 0.5 * SLOT_TIME * tf)
        nonreal_arrive = np.random.poisson(spot_means * 0.5 * SLOT_TIME * tf)

        for i in range(NUM_SPOTS):
            arr_r = real_arrive[i]
            arr_nr = nonreal_arrive[i]
            self.total_arrived[i] += (arr_r + arr_nr)
            self.nonreal_queues[i] += arr_nr
            for _ in range(arr_r):
                self.real_queues[i].append(self.slot_index)

        # 2. 激活波束
        active = np.where(action_vec == 1)[0]
        if len(active) == 0:
            active = np.random.choice(NUM_SPOTS, NUM_BEAMS, replace=False)

        # 3. 功率分配（公式 18）
        weights = {}
        for i in active:
            real_cnt = len(self.real_queues[i])
            nonreal_cnt = self.nonreal_queues[i]
            total_cnt = real_cnt + nonreal_cnt
            if real_cnt > 0:
                avg_delay = np.mean([(self.slot_index - p) * SLOT_TIME * 1000 for p in self.real_queues[i]])
            else:
                avg_delay = 1.0
            weights[i] = total_cnt * max(avg_delay, 1e-6)

        total_weight = sum(weights.values())
        power_allocation = {}
        if total_weight == 0:
            for i in active:
                power_allocation[i] = TOTAL_POWER / len(active)
        else:
            for i in active:
                power_allocation[i] = min(weights[i] / total_weight * TOTAL_POWER, MAX_BEAM_POWER)

        # 4. 容量与服务计算
        capacities = compute_capacity_fast(active, power_allocation)

        served_real_pkts = 0
        served_nonreal_pkts = 0

        for i in active:
            cap_bps = capacities[i]
            max_pkts_can_serve = int((cap_bps * SLOT_TIME) / PACKET_SIZE)

            # 先服务实时业务
            r_queue = self.real_queues[i]
            serve_r = min(len(r_queue), max_pkts_can_serve)
            for _ in range(serve_r):
                r_queue.popleft()

            self.total_served[i] += serve_r
            served_real_pkts += serve_r
            rem_cap = max_pkts_can_serve - serve_r

            # 再服务非实时业务
            if rem_cap > 0 and self.nonreal_queues[i] > 0:
                serve_nr = min(self.nonreal_queues[i], rem_cap)
                self.nonreal_queues[i] -= serve_nr
                self.total_served[i] += serve_nr
                served_nonreal_pkts += serve_nr

        self.total_served_nonreal_packets += served_nonreal_pkts

        # 5. 排队超时丢包处理
        for i in range(NUM_SPOTS):
            q = self.real_queues[i]
            while q:
                if (self.slot_index - q[0]) * SLOT_TIME * 1000.0 > MAX_DELAY:
                    q.popleft()
                else:
                    break

        # 6. 计算评估指标
        all_delays = []
        for i in range(NUM_SPOTS):
            for arr in self.real_queues[i]:
                all_delays.append((self.slot_index - arr) * SLOT_TIME * 1000.0)
        avg_delay = np.mean(all_delays) if len(all_delays) > 0 else 0.0

        throughput_mbps = (served_nonreal_pkts * PACKET_SIZE / 1e6) / SLOT_TIME
        mean_satisfaction = np.mean(self.satisfaction_vec)

        # 7. 多目标奖励定义 (公式 27-29)
        r1 = -avg_delay / 100.0
        r2 = throughput_mbps / 300.0
        r3 = mean_satisfaction

        self.history.append({
            'delay': avg_delay,
            'throughput': throughput_mbps,
            'satisfaction': mean_satisfaction
        })

        next_state = self._get_state()
        return next_state, (r1, r2, r3), False, {}

    def get_period_stats(self):
        if not self.history:
            return (0.0, 0.0, 0.0)
        delays = [h['delay'] for h in self.history]
        throughputs = [h['throughput'] for h in self.history]
        satisfactions = [h['satisfaction'] for h in self.history]
        return (float(np.mean(delays)), float(np.mean(throughputs)), float(np.mean(satisfactions)))

# ========================== 论文 CNN & FC 神经网络架构 (图13, 图14) ==========================
class DQN_Conv_Agent(nn.Module):
    """用于时延最小化和吞吐量最大化的卷积网络 (图13)"""
    def __init__(self, in_channels=2, action_dim=ACTION_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=(1, 20), padding=(0, 10))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 5), padding=(0, 2))
        self.fc1 = nn.Linear(16 * 12 * 41, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQN_FC_Agent(nn.Module):
    """用于业务满意度最大化的全连接网络 (图14)"""
    def __init__(self, in_dim=NUM_SPOTS, action_dim=ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ========================== MoE 多智能体 DQN 算法 ==========================
class MoE_MultiAgent_DQN:
    def __init__(self, lr=1e-5, gamma=0.9, epsilon=0.5, epsilon_min=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.999

        # 三个专长 Q 网络与目标网络
        self.q_net1 = DQN_Conv_Agent().to(DEVICE) # 时延
        self.q_net2 = DQN_Conv_Agent().to(DEVICE) # 吞吐量
        self.q_net3 = DQN_FC_Agent().to(DEVICE)   # 满意度

        self.target_net1 = DQN_Conv_Agent().to(DEVICE)
        self.target_net2 = DQN_Conv_Agent().to(DEVICE)
        self.target_net3 = DQN_FC_Agent().to(DEVICE)
        self.update_target(tau=1.0)

        self.opt1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.opt2 = optim.Adam(self.q_net2.parameters(), lr=lr)
        self.opt3 = optim.Adam(self.q_net3.parameters(), lr=lr)

        self.memory = deque(maxlen=3000)

    def update_target(self, tau=1.0):
        if tau == 1.0:
            self.target_net1.load_state_dict(self.q_net1.state_dict())
            self.target_net2.load_state_dict(self.q_net2.state_dict())
            self.target_net3.load_state_dict(self.q_net3.state_dict())

    def act(self, state, eval_mode=False):
        state_conv, state_sat = state
        if not eval_mode and np.random.random() < self.epsilon:
            idx = np.random.randint(ACTION_DIM)
        else:
            s_conv_t = torch.FloatTensor(state_conv).unsqueeze(0).to(DEVICE)
            s_sat_t = torch.FloatTensor(state_sat).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                q1 = self.q_net1(s_conv_t).squeeze(0)
                q2 = self.q_net2(s_conv_t).squeeze(0)
                q3 = self.q_net3(s_sat_t).squeeze(0)

                # L2 范数归一化 (论文 2.2.2)
                q1_norm = q1 / (torch.norm(q1, p=2) + 1e-8)
                q2_norm = q2 / (torch.norm(q2, p=2) + 1e-8)
                q3_norm = q3 / (torch.norm(q3, p=2) + 1e-8)

                # 线性标量化组合 (w1=w2=w3=1/3)
                q_total = (1/3.0) * q1_norm + (1/3.0) * q2_norm + (1/3.0) * q3_norm

            idx = torch.argmax(q_total).item()

        action_vec = np.zeros(NUM_SPOTS)
        action_vec[list(ALL_ACTIONS[idx])] = 1
        return action_vec, idx

    def remember(self, state, action_idx, rewards, next_state, done):
        self.memory.append((state, action_idx, rewards, next_state, done))

    def replay(self, batch_size=8):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)

        s_conv_b = torch.FloatTensor(np.array([b[0][0] for b in batch])).to(DEVICE)
        s_sat_b = torch.FloatTensor(np.array([b[0][1] for b in batch])).to(DEVICE)
        a_b = torch.LongTensor(np.array([b[1] for b in batch])).to(DEVICE)
        r1_b = torch.FloatTensor(np.array([b[2][0] for b in batch])).to(DEVICE)
        r2_b = torch.FloatTensor(np.array([b[2][1] for b in batch])).to(DEVICE)
        r3_b = torch.FloatTensor(np.array([b[2][2] for b in batch])).to(DEVICE)

        ns_conv_b = torch.FloatTensor(np.array([b[3][0] for b in batch])).to(DEVICE)
        ns_sat_b = torch.FloatTensor(np.array([b[3][1] for b in batch])).to(DEVICE)
        done_b = torch.BoolTensor(np.array([b[4] for b in batch])).to(DEVICE)

        # 训练 3 个网络
        for q_net, target_net, opt, s_b, ns_b, r_b in [
            (self.q_net1, self.target_net1, self.opt1, s_conv_b, ns_conv_b, r1_b),
            (self.q_net2, self.target_net2, self.opt2, s_conv_b, ns_conv_b, r2_b),
            (self.q_net3, self.target_net3, self.opt3, s_sat_b, ns_sat_b, r3_b)
        ]:
            curr_q = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            next_q = target_net(ns_b).max(1)[0].detach()
            target_q = r_b + self.gamma * next_q * (~done_b)

            loss = nn.MSELoss()(curr_q, target_q)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ========================== 仿真主程序与数据输出 ==========================
def run_simulation():
    print(f"使用计算设备: {DEVICE}")
    env = BeamHoppingEnv()
    agent = MoE_MultiAgent_DQN()

    LOOPS = 450
    TIME_SLOTS = 1000
    BATCH_SIZE = 8
    TARGET_UPDATE = 100

    # 存储每一个训练周期的结果元组
    all_episode_stats = []

    print("\n>>> 开始多智能体强化学习模型训练 (450 周期) <<<")
    for episode in range(LOOPS):
        state = env.reset()
        for t in range(TIME_SLOTS):
            action_vec, action_idx = agent.act(state)
            next_state, rewards, done, _ = env.step(action_vec)

            agent.remember(state, action_idx, rewards, next_state, done)
            agent.replay(BATCH_SIZE)

            state = next_state
            if t % TARGET_UPDATE == 0:
                agent.update_target(tau=1.0)

        # 统计每个训练周期数据 (平均时延, 吞吐量, 满意度)
        period_stat = env.get_period_stats()
        all_episode_stats.append(period_stat)
        agent.decay_epsilon()

        if (episode + 1) % 50 == 0 or episode == 0:
            print(f"Episode {episode + 1:03d}/{LOOPS} | 平均时延: {period_stat[0]:.2f} ms | 吞吐量: {period_stat[1]:.2f} Mbps | 满意度: {period_stat[2]:.4f}")

    # 将训练周期结果整理为元组集合的形式输出
    tuple_dataset_episodes = set(all_episode_stats)

    print("\n>>> 开始时变环境下算法性能评估 (3.2 仿真结果与分析) <<<")
    eval_env = BeamHoppingEnv()
    state = eval_env.reset()
    time_varying_stats = []

    for t in range(200):
        action_vec, _ = agent.act(state, eval_mode=True)
        next_state, _, done, _ = eval_env.step(action_vec)

        slot_delay = eval_env.history[-1]['delay']
        slot_tp = eval_env.history[-1]['throughput']
        slot_sat = eval_env.history[-1]['satisfaction']

        time_varying_stats.append((slot_delay, slot_tp, slot_sat))
        state = next_state

    # 汇总时变环境元组集合
    tuple_dataset_time_varying = set(time_varying_stats)

    print("\n======================== 结果输出汇总 ========================")
    print(f"1. 训练周期元组集合 (共 {len(tuple_dataset_episodes)} 项):")
    print("前 5 个周期的结果元组示例 (平均时延 ms, 吞吐量 Mbps, 满意度):")
    for item in all_episode_stats[:5]:
        print(f"   {item}")

    print("\n2. 时变环境下性能评估元组集合 (前 10 个时隙):")
    for idx, item in enumerate(time_varying_stats[:10]):
        print(f"   Slot {idx+1:03d}: 时延={item[0]:.2f}ms, 吞吐量={item[1]:.2f}Mbps, 满意度={item[2]:.4f}")

    return tuple_dataset_episodes, tuple_dataset_time_varying


episodes_set, time_varying_set = run_simulation()