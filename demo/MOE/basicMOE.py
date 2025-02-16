import torch
import torch.nn as nn


class BasicExpert(nn.Module):
    # 一个 Expert 可以是一个最简单的， linear 层即可
    # 也可以是 MLP 层
    # 也可以是 更复杂的 MLP 层（active function 设置为 swiglu）
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)

    def forward(self, x):
        return self.linear(x)


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

        expert_weight = expert_weight.unsqueeze(1)  # (batch, 1, expert_nuber)

        # expert_weight * expert_out_list
        output = expert_weight @ expert_output  # (batch, 1, feature_out)

        return output.squeeze()


def test_basic_moe():
    x = torch.rand(2, 4)

    basic_moe = BasicMOE(4, 3, 2)
    out = basic_moe(x)
    print(out)


test_basic_moe()