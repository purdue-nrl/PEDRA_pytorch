import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot, make_dot_from_trace
from network.communication import communication_setup
from configs.read_cfg import read_cfg, update_algorithm_cfg


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0),-1)
        #print(x.size())
        x = self.fc2(self.fc1(x))
        return x
    
class PedraAgent():
    def __init__(self, vehicle_name,  device):
        self.vehicle_name = vehicle_name
        self.device = device
        # torch.manual_seed(1234)
        # torch.cuda.manual_seed(1234)
        self.policy =  Net().to(device)
        self.optimizer =  torch.optim.Adam(self.policy.parameters(), lr=0.1)
        self.input_tensor = torch.randn(16,1,3,3).to(device)
        target = torch.randn(16).to(device)  # a dummy target, for example
        self.target = target.view(1, -1)  # make it the same shape as output
        self.criterion = nn.MSELoss()
        
#####################
device_list = {}
agent = {}
name_agent_list = []

algorithm_cfg = read_cfg(config_filename='configs/actorcritic_dist.cfg', verbose=True)

for drone in range(4):
    name_agent = str(drone)
    device_list[name_agent] = torch.device("cuda:{}".format(drone))    
    name_agent_list.append(name_agent)
    agent[name_agent] = PedraAgent(vehicle_name=name_agent, device=device_list[name_agent])
    
comm_setup = communication_setup(agent, name_agent_list, algorithm_cfg, device_list)

for i in range(0,10):
    for name in range(0,4):
        name_agent = str(name)
        agent[name_agent].optimizer.zero_grad()
        output = agent[name_agent].policy(agent[name_agent].input_tensor)
        loss = agent[name_agent].criterion(output, agent[name_agent].target)
        loss.backward()
        agent[name_agent].optimizer.step()
        for p in agent[name_agent].policy.parameters():
            print(name_agent, p[0]); break
        
    comm_setup.gossip(agent)
    for name in range(0,4):
        name_agent = str(name)
        for p in agent[name_agent].policy.parameters():
            print(name_agent, p[0]); break
            