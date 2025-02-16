1.MOE原理以及手撕学习

MoE架构的主要特点是在模型中引入了专家网络层，通过**路由机制(Routing function)**选择激活哪些专家，以允许不同的专家模型对输入进行独立处理，并通过**加权组合它们的输出**来生成最终的预测结果。

“*MoE提出的前提是如果有一个包括了多个领域知识的复杂问题，我们该使用什么样的方法来解决呢？最简单的办法就是把各个领域的专家集合到一起来攻克这个任务，当然我们事先要把[不同的](https://so.csdn.net/so/search?q=不同的&spm=1001.2101.3001.7020)任务先分离出来，这样才便于分发给不同领域的专家，让他们来帮忙处理，最后再汇总结论。*”这也是大模型能实现大规模推理的原因之一，每次只激活部分参数。

1.1基础MOE demo

最基础的MOE实际上就是用 线形层等对输入计算出expert的概率分布，再用每个expert对输出进行计算，再按概率分布进行加权。

<img src="C:\Users\Dunker\AppData\Roaming\Typora\typora-user-images\image-20250216114016756.png" alt="image-20250216114016756" style="zoom:50%;" />

```python

class BasicMOE(nn.Module):
    def __init__(self, feature_in, feature_out, expert_number):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                BasicExpert(feature_in, feature_out) for _ in range(expert_number)
            ]
        )
        # gate 就是选一个 expert 
        self.gate = nn.Linear(feature_in, expert_number)
        
    def forward(self, x):
        # x 的 shape 是 （batch, feature_in)
        expert_weight = self.gate(x)  # shape 是 (batch, expert_number)
        expert_out_list = [
            expert(x).unsqueeze(1) for expert in self.experts
        ]  # 里面每一个元素的 shape 是： (batch, ) ??
        # concat 起来 (batch, expert_number, feature_out)
        expert_output = torch.cat(expert_out_list, dim=1)
        # print(expert_output.size())
        expert_weight = expert_weight.unsqueeze(1) # (batch, 1, expert_number)
        # expert_weight * expert_out_list
        output = expert_weight @ expert_output  # (batch, 1, feature_out)
        
        return output.squeeze()
    
def test_basic_moe():
    x = torch.rand(2, 4)
    basic_moe = BasicMOE(4, 3, 2)
    out = basic_moe(x)
    print(out)
    
test_basic_moe()
```

1.2MOE plus版：[SparseMoE ]

<img src="C:\Users\Dunker\AppData\Roaming\Typora\typora-user-images\image-20250216114630287.png" alt="image-20250216114630287" style="zoom:50%;" />

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    # 一个 Expert 可以是一个最简单的， linear 层即可
    # 也可以是 MLP 层
    # 也可以是 更复杂的 MLP 层（active function 设置为 swiglu）
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)
    def forward(self, x):
        return self.linear(x)
class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k
    def forward(self, hidden_states):
        # 计算路由logits
        router_logits = self.gate(hidden_states)  # shape is (b * s, expert_number)

        # 计算专家经过softmax之后的概率
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # 计算topk的专家的输出
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  # shape都是 (b * s, top_k)

        # 专家权重归一化
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)

        # 生成专家掩码
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )  # shape是 (b * s, top_k, expert_number)
        expert_mask = expert_mask.permute(2, 1, 0)  # (expert_number, top_k, b * s)

        return router_logits, router_weights, selected_experts, expert_mask
class MOEConfig:
    def __init__(
            self,
            hidden_dim,
            expert_number,
            top_k,
            shared_experts_number=2,
    ):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number
class SparseMOE(nn.Module):
    # 稀疏 MOE 模型，这里每一个 token 都会过 topk 个专家，得到对应token 的 hidden_embeddings
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number
        self.top_k = config.top_k
        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_number)
            ]
        )
        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)
    def forward(self, x):
        # x shape is (b, s, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()  # 2 4 16
        # 合并前两个维度，因为不是 Sample 维度了，而是 token 维度
        hidden_states = x.view(-1, hidden_dim)  # shape is(b * s, hidden_dim)

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        # 其中 selected_experts_indices shape 是 (b * s, top_k)
        # 其中 expert_mask shape 是 (expert_number, top_k, b * s)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape 是 (top_k, b * s)
            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx 和 top_x 都是一维 tensor
            # idx 的值是 0 或 1, 表示这个 token 是作为当前专家的 top1 还是 top2
            # top_x 的值是 token 在 batch*seq_len 中的位置索引
            # 例如对于 batch_size=2, seq_len=4 的输入:
            # top_x 的值范围是 0-7, 表示在展平后的 8 个 token 中的位置
            # idx 的值是 0/1, 表示这个 token 把当前专家作为其 top1/top2 专家

            # hidden_states 的 shape 是 (b * s, hidden_dim)
            # 需要取到 top_x 对应的 hidden_states
            current_state = hidden_states.unsqueeze(
                0
            )[:, top_x, :].reshape(-1, hidden_dim)  # （selected_token_number, hidden_dim）

            # router_weight 的 shape 是 (b * s, top_k)
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[top_x, idx].unsqueeze(-1)  # （selected_token_number, 1） 这里有广播
            # 把当前专家的输出加到 final_hidden_states 中
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        # 把 final_hidden_states 还原到原来的 shape
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits  # shape 是 (b * s, expert_number)
def test_token_level_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    token_level_moe = SparseMOE(config)
    out = token_level_moe(x)
    print(out[0].shape, out[1].shape)

test_token_level_moe()
```

1.3 deepseek 版MOE

相较普通的多了一个 shared experts 模型，所有 token 都过这个 shared experts 模型，然后每个 token 会用计算的 Router 权重，来选择 topK 个专家，然后和共享的专家的输出一起加权求和。

<img src="C:\Users\Dunker\AppData\Roaming\Typora\typora-user-images\image-20250216192421393.png" alt="image-20250216192421393" style="zoom:50%;" />

deepseek的moe 中share experts相当于提供通用的特征提取，与“专精”的expert做加权

1.4  既然只选了top k个expert,如何反向传播？豆包的回答：按照前向时的权重算出对应expert的梯度

![image-20250216193555699](C:\Users\Dunker\AppData\Roaming\Typora\typora-user-images\image-20250216193555699.png)

1.5 MMOE

**MMOE**(Multi-gate Mixture-of-Experts)是在MOE的基础上，使用了多个门控网络， ![k](https://www.zhihu.com/equation?tex=k&consumer=ZHI_MENG) 个任就对应 ![k](https://www.zhihu.com/equation?tex=k&consumer=ZHI_MENG) 个门控网络。所以在多任务时可以考虑这个MMOE。

“相对于 **MOE**的结构中所有任务共享一个门控网络，**MMOE**的结构优化为每个任务都单独使用一个门控网络。这样的改进可以针对不同任务得到不同的 Experts 权重，从而实现对 Experts 的选择性利用，不同任务对应的门控网络可以学习到不同的Experts 组合模式，因此模型更容易捕捉到子任务间的相关性和差异性。”

<img src="C:\Users\Dunker\AppData\Roaming\Typora\typora-user-images\image-20250216220659093.png" alt="image-20250216220659093" style="zoom:50%;" />

2.双塔网络 

 ![image-20250216220041771](C:\Users\Dunker\AppData\Roaming\Typora\typora-user-images\image-20250216220041771.png)



2.强化学习与手撕

2.1原理

“强化学习通过与环境的交互获得反馈，不断更新策略，以达到最优的决策目标。在推荐系统中，RL可以通过实时学习用户的反馈（如点击、浏览、购买等），动态调整推荐策略，从而提升推荐效果。”

![image-20250216234937292](C:\Users\Dunker\AppData\Roaming\Typora\typora-user-images\image-20250216234937292.png)

![image-20250216235214976](C:\Users\Dunker\AppData\Roaming\Typora\typora-user-images\image-20250216235214976.png)

2.2 DQN

强化学习在推荐系统中的核心算法主要包括：

Q-Learning：基于Q值的强化学习算法，用于学习代理在环境中实现最佳行为。
Deep Q-Network(DQN)：基于深度神经网络的Q-Learning算法，用于解决推荐系统中的数据稀疏性问题。
Policy Gradient：基于策略梯度的强化学习算法，用于学习多种策略以适应各种不同的用户需求。

Deep Q-Network(DQN)是基于深度神经网络的Q-Learning算法，它可以解决推荐系统中的数据稀疏性问题。DQN的核心概念包括：

①深度神经网络：用于估计Q值的神经网络

②经验重放 Buffer：用于存储经验的缓冲区

③目标网络：用于学习最优策略的目标神经网络

豆包生成代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 定义 DQN 网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        # 前向传播
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义推荐环境
class RecommendationEnv:
    def __init__(self, num_items):
        self.num_items = num_items
        self.reset()

    def reset(self):
        # 重置环境，返回初始状态
        self.user_state = np.random.randn(10)  # 随机初始化用户状态
        return self.user_state

    def step(self, action):
        # 执行动作，返回下一个状态、奖励和是否结束的标志
        # 简单模拟奖励，假设某些物品更受用户欢迎
        reward = np.random.randn() if action % 2 == 0 else -np.random.randn()
        self.user_state = np.random.randn(10)  # 更新用户状态
        done = False  # 简单假设不会结束
        return self.user_state, reward, done

# 定义 DQN 代理
class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995  # 探索率衰减率
        self.epsilon_min = 0.01  # 最小探索率
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.model = DQN(input_dim, output_dim)  # 主网络
        self.target_model = DQN(input_dim, output_dim)  # 目标网络
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # 将经验存储到回放缓冲区
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 根据当前状态选择动作
        if np.random.rand() <= self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.output_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        # 利用：选择 Q 值最大的动作
        action = torch.argmax(q_values).item()
        return action

    def replay(self, batch_size):
        # 经验回放
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = self.model(state).clone()
            if done:
                target[0][action] = reward
            else:
                t = self.target_model(next_state).detach()
                target[0][action] = reward + self.gamma * torch.max(t).item()
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        # 更新目标网络
        self.target_model.load_state_dict(self.model.state_dict())

# 主训练循环
if __name__ == "__main__":
    num_items = 10  # 物品数量
    env = RecommendationEnv(num_items)
    agent = DQNAgent(input_dim=10, output_dim=num_items)
    batch_size = 32
    episodes = 100

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(50):  # 每个回合最多执行 50 步
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
            total_reward += reward
        agent.update_target_model()
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
```





Sparse Enhanced Network: An Adversarial Generation Method for Robust Augmentation in Sequential Recommendation---From AAAI2024

关注的是稀疏问题，







KDD2024-Trinity: Syncretizing Multi-/Long-tail/Long-term Interests All in One







 WWW2024-Collaborative Large Language Model for Recommender Systems





