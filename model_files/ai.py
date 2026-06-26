import torch
import torch.nn as nn
import copy
import random

class BlockBlastAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 5 * 5 + 64 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 64)
        )
    def forward(self, lst):
        tensor = torch.tensor(lst, dtype=torch.float32)
        return self.model(tensor).tolist()
    def mutate_model(self):
        child_model = copy.deepcopy(self)
        # if generation > 300:
        if random.randint(0, 2):
            mutation_rate = 0.05
        else:
            mutation_rate = 0.2
        if random.randint(0, 2):
            mutation_power = 0.1 #* (1 - (1 / 12500) * generation)
        else:
            mutation_power = 0.5
        # else:
        #     mutation_rate = 0.035
        #     mutation_power = 0.35
        # if generation < 2000:
        #     mutation_rate = 0.25
        #     mutation_power = 1
        with torch.no_grad():
            for param in child_model.parameters():
                mutation_mask = (torch.rand_like(param) < mutation_rate).float()
                noise = torch.randn_like(param) * mutation_power * mutation_mask
                param.add_(noise)
        return child_model

class Heuristic_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.final_net = nn.Sequential(
            nn.Linear(128 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, board, other_info):
        return self.final_net(torch.cat((self.cnn(board), other_info), dim=1))

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.board_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten() 
        )
        self.flat_fc = nn.Sequential(
            nn.Linear(77, 128),
            nn.ReLU()
        )
        self.final_net = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 192)
        )
    def forward(self, x):
        pieces = x[:, :75]
        board_flat = x[:, 75:139]  
        combos = x[:, 139:]
        board_2d = board_flat.view(-1, 1, 8, 8)
        flat_data = torch.cat([pieces, combos], dim=1)
        board_features = self.board_conv(board_2d)
        flat_features = self.flat_fc(flat_data)
        combined_features = torch.cat([board_features, flat_features], dim=1)
        q_values = self.final_net(combined_features)
        return q_values