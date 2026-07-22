import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from itertools import combinations
import os
import pickle
import sys

# ========================== 固定随机种子 ==========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ========================== 系统参数（表1） ==========================
NUM_SPOTS = 12
NUM_BEAMS = 4
SLOT_TIME = 0.01          # 10 ms
MAX_DELAY = 400           # ms
PACKET_SIZE = 10 * 1024 * 8  # 10 kbit → bit
BANDWIDTH = 200e6         # 200 MHz
TOTAL_POWER = 120         # W
MAX_BEAM_POWER = 60       # W
ORBIT_ALTITUDE = 570e3    # 570 km
FREQ = 20e9               # 20 GHz
C = 3e8
LAMBDA = C / FREQ
G_TX = 40                 # dB -> 线性 10000
G_RX = 50                 # dB -> 线性 100000
K_B = 1.38e-23
TEMP = 300
NOISE_POWER = K_B * TEMP * BANDWIDTH  # W

# 波位位置：覆盖半径 500 km
COVERAGE_RADIUS = 500e3
spot_positions = np.random.uniform(-COVERAGE_RADIUS, COVERAGE_RADIUS, (NUM_SPOTS, 2))
sat_pos = np.array([0, 0, ORBIT_ALTITUDE])

# 预生成所有组合动作 C(12,4)=495
ALL_ACTIONS = list(combinations(range(NUM_SPOTS), NUM_BEAMS))
ACTION_DIM = len(ALL_ACTIONS)

# ========================== 业务模型（无 scipy） ==========================
# 生成截断正态分布的波位峰值速率（均值150，标准差75，截断在[50,300]）
mean_traffic = 150
std = 75
# 用numpy生成正态，然后截断
raw = np.random.normal(mean_traffic, std, NUM_SPOTS)
spot_means = np.clip(raw, 50, 300)   # 保证在50~300之间

def time_factor_hour(hour):
    """图7 时间加权因子"""
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

# ========================== 干扰与容量函数（公式2-7） ==========================
def antenna_gain(theta, theta_3dB=0.2):
    """贝塞尔函数近似（无需scipy，直接用numpy的jv）"""
    # 用numpy的jv函数（numpy也有jv，但为了保险，我们使用math或scipy？）
    # 这里使用numpy的vectorize，但需要导入numpy的jv
    try:
        from numpy import jv
    except ImportError:
        # 若numpy没有jv，则回退到scipy
        from scipy.special import jv
    if theta == 0:
        return 1.0
    u = 2.07123 * np.sin(theta) / np.sin(theta_3dB)
    J1 = jv(1, u)
    J3 = jv(3, u)
    G = (J1 / (2*u) + 36 * J3 / (u**3))**2
    return G * 10000

def compute_interference(active_indices, sat_pos, spot_positions, power_allocation):
    num_spots = len(spot_positions)
    interference = np.zeros(num_spots)
    for i in active_indices:
        P_i = power_allocation[i]
        pos_i = spot_positions[i]
        d_i = np.linalg.norm(pos_i - sat_pos[:2])
        dist_i = np.sqrt(d_i**2 + sat_pos[2]**2)
        for j in range(num_spots):
            if i == j:
                continue
            pos_j = spot_positions[j]
            d_j = np.linalg.norm(pos_j - sat_pos[:2])
            dist_j = np.sqrt(d_j**2 + sat_pos[2]**2)
            d_ij = np.linalg.norm(pos_i - pos_j)
            if d_ij == 0:
                continue
            cos_theta = (d_i**2 + d_j**2 + 2*sat_pos[2]**2 - d_ij**2) / (2 * np.sqrt(d_i**2 + sat_pos[2]**2) * np.sqrt(d_j**2 + sat_pos[2]**2))
            theta = np.arccos(np.clip(cos_theta, -1, 1))
            G_theta = antenna_gain(theta)
            interference[j] += (G_TX * P_i * G_theta * LAMBDA**2) / ((4 * np.pi * d_ij)**2)
    return interference

def compute_capacity(active_indices, sat_pos, spot_positions, power_allocation):
    capacities = np.zeros(NUM_SPOTS)
    interference = compute_interference(active_indices, sat_pos, spot_positions, power_allocation)
    for i in active_indices:
        d_i = np.linalg.norm(spot_positions[i] - sat_pos[:2])
        dist_i = np.sqrt(d_i**2 + sat_pos[2]**2)
        path_loss = (4 * np.pi * dist_i / LAMBDA)**2
        signal = G_TX * power_allocation[i] * G_RX / path_loss
        sinr = signal / (interference[i] + NOISE_POWER + 1e-12)
        capacities[i] = BANDWIDTH * np.log2(1 + sinr) / 1e6   # Mbps
    return capacities

# ========================== 环境类 ==========================
class BeamHoppingEnv:
    def __init__(self, train_mode=True):
        self.train_mode = train_mode
        self.history_len = 40
        self.reset()

    def reset(self):
        self.real_queue = deque()
        self.nonreal_queue = np.zeros(NUM_SPOTS, dtype=float)
        self.total_real_arrived = np.zeros(NUM_SPOTS, dtype=float)
        self.total_nonreal_arrived = np.zeros(NUM_SPOTS, dtype=float)
        self.served_real = np.zeros(NUM_SPOTS, dtype=float)
        self.served_nonreal = np.zeros(NUM_SPOTS, dtype=float)
        self.slot_index = 0
        self.history = []
        self.state_history_real = np.zeros((NUM_SPOTS, self.history_len))
        self.state_history_nonreal = np.zeros((NUM_SPOTS, self.history_len))
        return self._get_state()

    def _get_state(self):
        # 当前队列长度（包数）
        real_len = np.zeros(NUM_SPOTS)
        for _, spot in self.real_queue:
            real_len[spot] += 1
        nonreal_len = self.nonreal_queue.copy()  # Mbit，但转为包数（除以0.01）
        nonreal_len = nonreal_len / 0.01   # 折合包数
        # 更新历史矩阵
        self.state_history_real = np.roll(self.state_history_real, -1, axis=1)
        self.state_history_nonreal = np.roll(self.state_history_nonreal, -1, axis=1)
        self.state_history_real[:, -1] = real_len
        self.state_history_nonreal[:, -1] = nonreal_len
        state = np.stack([self.state_history_real, self.state_history_nonreal], axis=0)
        return state.astype(np.float32)

    def step(self, action_vec):
        self.slot_index += 1
        hour = 9 + self.slot_index * SLOT_TIME / 3600
        tf = time_factor_hour(hour)

        # 1. 业务到达 (Mbit)
        real_arrive = np.random.poisson(spot_means * 0.5 * SLOT_TIME * tf)
        nonreal_arrive = np.random.poisson(spot_means * 0.5 * SLOT_TIME * tf)
        self.total_real_arrived += real_arrive
        self.total_nonreal_arrived += nonreal_arrive
        self.nonreal_queue += nonreal_arrive

        for spot_idx, packets in enumerate(real_arrive):
            for _ in range(int(packets)):
                self.real_queue.append((self.slot_index, spot_idx))

        # 2. 激活波束
        active = np.where(action_vec == 1)[0]
        if len(active) == 0:
            active = np.random.choice(NUM_SPOTS, NUM_BEAMS, replace=False)

        # 3. 功率分配 (公式18)
        weights = {}
        for i in active:
            real_cnt = sum(1 for p in self.real_queue if p[1] == i)
            nonreal_cnt = self.nonreal_queue[i] / 0.01   # 包数
            total_packets = real_cnt + nonreal_cnt
            delays = []
            for p in self.real_queue:
                if p[1] == i:
                    delays.append((self.slot_index - p[0]) * SLOT_TIME * 1000)
            avg_delay = np.mean(delays) if delays else 1.0
            weights[i] = total_packets * max(avg_delay, 1e-6)
        total_weight = sum(weights.values())
        if total_weight == 0:
            power_allocation = {i: TOTAL_POWER / len(active) for i in active}
        else:
            power_allocation = {}
            for i in active:
                power_allocation[i] = weights[i] / total_weight * TOTAL_POWER
                power_allocation[i] = min(power_allocation[i], MAX_BEAM_POWER)

        # 4. 容量计算
        capacities = compute_capacity(active, sat_pos, spot_positions, power_allocation)

        # 5. 服务数据
        served_real = np.zeros(NUM_SPOTS)
        served_nonreal = np.zeros(NUM_SPOTS)
        for i in active:
            cap_mbps = capacities[i]
            max_serve_mbits = cap_mbps * SLOT_TIME
            # 服务实时
            remaining = max_serve_mbits
            temp_queue = deque()
            while self.real_queue and remaining > 0:
                arrival, spot = self.real_queue.popleft()
                if spot == i:
                    if remaining >= 0.01:
                        remaining -= 0.01
                        served_real[i] += 0.01
                    else:
                        self.real_queue.appendleft((arrival, spot))
                        break
                else:
                    temp_queue.append((arrival, spot))
            while temp_queue:
                self.real_queue.appendleft(temp_queue.pop())
            # 服务非实时
            if remaining > 0 and self.nonreal_queue[i] > 0:
                serve = min(self.nonreal_queue[i], remaining)
                self.nonreal_queue[i] -= serve
                served_nonreal[i] = serve

        self.served_real += served_real
        self.served_nonreal += served_nonreal

        # 6. 超时丢弃
        while self.real_queue:
            arrival, _ = self.real_queue[0]
            if (self.slot_index - arrival) * SLOT_TIME * 1000 > MAX_DELAY:
                self.real_queue.popleft()
            else:
                break

        # 7. 指标
        if self.real_queue:
            delays = [(self.slot_index - arrival) * SLOT_TIME * 1000 for arrival, _ in self.real_queue]
            avg_delay = np.mean(delays)
        else:
            avg_delay = 0.0

        total_served_nonreal = served_nonreal.sum()
        throughput_mbps = total_served_nonreal / SLOT_TIME if SLOT_TIME > 0 else 0

        total_req = self.total_real_arrived.sum() + self.total_nonreal_arrived.sum()
        if total_req > 0:
            satisfaction = (self.served_real.sum() + self.served_nonreal.sum()) / total_req
        else:
            satisfaction = 0.5

        # 奖励
        r1 = -avg_delay / 100.0
        r2 = throughput_mbps / 400.0
        r3 = satisfaction
        reward = 0.3 * r1 + 0.3 * r2 + 0.4 * r3

        self.history.append({
            'delay': avg_delay,
            'throughput_mbps': throughput_mbps,
            'satisfaction': satisfaction,
            'reward': reward
        })

        next_state = self._get_state()
        return next_state, reward, False, {}

    def get_period_stats(self):
        if not self.history:
            return 0, 0, 0
        return (np.mean([h['delay'] for h in self.history]),
                np.mean([h['throughput_mbps'] for h in self.history]),
                np.mean([h['satisfaction'] for h in self.history]))

# ========================== DQN 网络 (CNN) ==========================
class DQN_CNN(nn.Module):
    def __init__(self, input_channels=2, action_dim=ACTION_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 12 * 40, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ========================== 多目标 DQN ==========================
class MultiObjectiveDQN:
    def __init__(self, state_shape, lr=1e-5, gamma=0.9, epsilon=0.5, epsilon_min=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.999

        self.q_net1 = DQN_CNN(input_channels=2, action_dim=ACTION_DIM)
        self.q_net2 = DQN_CNN(input_channels=2, action_dim=ACTION_DIM)
        self.q_net3 = DQN_CNN(input_channels=2, action_dim=ACTION_DIM)
        self.target_net1 = DQN_CNN(input_channels=2, action_dim=ACTION_DIM)
        self.target_net2 = DQN_CNN(input_channels=2, action_dim=ACTION_DIM)
        self.target_net3 = DQN_CNN(input_channels=2, action_dim=ACTION_DIM)
        self.update_target(tau=1.0)

        self.optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)
        self.optimizer3 = optim.Adam(self.q_net3.parameters(), lr=lr)

        self.memory1 = deque(maxlen=3000)
        self.memory2 = deque(maxlen=3000)
        self.memory3 = deque(maxlen=3000)

    def update_target(self, tau=1.0):
        if tau == 1.0:
            self.target_net1.load_state_dict(self.q_net1.state_dict())
            self.target_net2.load_state_dict(self.q_net2.state_dict())
            self.target_net3.load_state_dict(self.q_net3.state_dict())

    def act(self, state, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            idx = np.random.randint(ACTION_DIM)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q1 = self.q_net1(state_t).squeeze(0)
                q2 = self.q_net2(state_t).squeeze(0)
                q3 = self.q_net3(state_t).squeeze(0)
                q1 = q1 / (torch.norm(q1, p=2) + 1e-8)
                q2 = q2 / (torch.norm(q2, p=2) + 1e-8)
                q3 = q3 / (torch.norm(q3, p=2) + 1e-8)
                q_total = q1 + q2 + q3
            idx = torch.argmax(q_total).item()
        action_vec = np.zeros(NUM_SPOTS)
        action_vec[list(ALL_ACTIONS[idx])] = 1
        return action_vec, idx

    def remember(self, state, action_idx, reward, next_state, done, target_idx):
        if target_idx == 1:
            self.memory1.append((state.copy(), action_idx, reward, next_state.copy(), done))
        elif target_idx == 2:
            self.memory2.append((state.copy(), action_idx, reward, next_state.copy(), done))
        else:
            self.memory3.append((state.copy(), action_idx, reward, next_state.copy(), done))

    def replay(self, batch_size, target_idx):
        if target_idx == 1:
            memory = self.memory1
            q_net = self.q_net1
            target_net = self.target_net1
            optimizer = self.optimizer1
        elif target_idx == 2:
            memory = self.memory2
            q_net = self.q_net2
            target_net = self.target_net2
            optimizer = self.optimizer2
        else:
            memory = self.memory3
            q_net = self.q_net3
            target_net = self.target_net3
            optimizer = self.optimizer3

        if len(memory) < batch_size:
            return
        batch = random.sample(memory, batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = torch.LongTensor(np.array([b[1] for b in batch]))
        rewards = torch.FloatTensor(np.array([b[2] for b in batch]))
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.BoolTensor(np.array([b[4] for b in batch]))

        current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = target_net(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (~dones)
        loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ========================== 训练主函数 ==========================
def train():
    try:
        print("程序开始运行...")
        env = BeamHoppingEnv(train_mode=True)
        state_shape = (2, NUM_SPOTS, 40)
        agent = MultiObjectiveDQN(state_shape)

        LOOPS = 450
        TIME_SLOTS = 1000
        BATCH_SIZE = 8
        TARGET_UPDATE = 100

        all_episode_stats = []
        for episode in range(LOOPS):
            state = env.reset()
            for t in range(TIME_SLOTS):
                action_vec, action_idx = agent.act(state)
                next_state, reward, done, _ = env.step(action_vec)
                agent.remember(state, action_idx, reward, next_state, done, 1)
                agent.remember(state, action_idx, reward, next_state, done, 2)
                agent.remember(state, action_idx, reward, next_state, done, 3)
                agent.replay(BATCH_SIZE, 1)
                agent.replay(BATCH_SIZE, 2)
                agent.replay(BATCH_SIZE, 3)
                state = next_state
                if t % TARGET_UPDATE == 0:
                    agent.update_target(tau=1.0)

            avg_delay, avg_throughput, avg_satisfaction = env.get_period_stats()
            all_episode_stats.append((avg_delay, avg_throughput, avg_satisfaction))
            agent.decay_epsilon()
            print(f"Episode {episode+1}/{LOOPS}, Delay={avg_delay:.2f}ms, Throughput={avg_throughput:.2f}Mbps, Satisfaction={avg_satisfaction:.4f}")

        with open('episode_stats.pkl', 'wb') as f:
            pickle.dump(all_episode_stats, f)
        print("训练数据已保存到 episode_stats.pkl")

        # 时变评估
        print("\n=== 时变环境性能评估 ===")
        eval_env = BeamHoppingEnv(train_mode=False)
        state = eval_env.reset()
        time_varying = []
        for t in range(200):
            action_vec, _ = agent.act(state, eval_mode=True)
            next_state, reward, done, _ = eval_env.step(action_vec)
            delay = eval_env.history[-1]['delay'] if eval_env.history else 0
            throughput = eval_env.history[-1]['throughput_mbps'] if eval_env.history else 0
            satisfaction = eval_env.history[-1]['satisfaction'] if eval_env.history else 0
            time_varying.append((delay, throughput, satisfaction))
            state = next_state
        with open('time_varying.pkl', 'wb') as f:
            pickle.dump(time_varying, f)
        print("时变数据已保存到 time_varying.pkl")
        print("前20个时隙数据：")
        for i, (d, t, s) in enumerate(time_varying[:20]):
            print(f"时隙{i+1}: Delay={d:.2f}ms, Throughput={t:.2f}Mbps, Satisfaction={s:.4f}")

    except Exception as e:
        print("程序发生异常：", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()