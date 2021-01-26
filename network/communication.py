# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:20:35 2021

@author: aparna
"""
import numpy as np
import torch
from network.compressor import *

class communication_setup():
    def __init__(self, agent, name_agent_list, cfg, device_list):
        self.world_size      = len(name_agent_list)
        self.graph_type      = cfg.graph
        self.weights_type    = cfg.weights
        self.name_agent_list = name_agent_list
        self.device_list     = device_list
        self.graph_connect   = self.create_graph()
        self.weights         = self.mixing_matrix()
        self.gossip_error    = self.set_buffer(agent)
        self.gossip_var      = self.set_buffer(agent)
        # define hyperparameters
        first_param_dtype   = next(agent[name_agent_list[0]].policy.parameters()).dtype
        self.averaging_rate = self.dist_scalar(cfg.eta, first_param_dtype)
        self.qlevel         = cfg.qlevel
        self.bias           = cfg.bias
        self.ratio          = cfg.ratio
        self.compressor     = cfg.compressor
        print('Graph  ----> ', self.graph_connect)
        print('Mixing ----> ', self.weights)
        

    def create_graph(self):
        if 'undirected_ring' in self.graph_type:
            connectivity = {}
            for i in range(0, self.world_size):
                forward_neigh  = (i+1) % self.world_size
                backward_neigh = (i-1) % self.world_size
                connectivity[self.name_agent_list[i]] = [self.name_agent_list[i],
                                                         self.name_agent_list[forward_neigh], 
                                                         self.name_agent_list[backward_neigh]] # list of outgoing neighbours
        else:
            raise NotImplementedError   
        return connectivity
    
    
    def mixing_matrix(self):
        if 'uniform' in self.weights_type:
            weights = {}
            for i in range(0, self.world_size):
                degree = len(self.graph_connect[self.name_agent_list[i]])
                weight = [1.0/float(degree)] * degree 
                weights[self.name_agent_list[i]] = torch.tensor(weight, device=self.device_list[self.name_agent_list[i]])
        else:
            raise NotImplementedError
        return weights
    
    def set_buffer(self, agent):
        buffer = {}
        for i in range(0, self.world_size):
            buffer[self.name_agent_list[i]] = []
            for p in agent[self.name_agent_list[0]].policy.parameters():
                cp = p.clone().detach_()
                cp = cp.to(self.device_list[self.name_agent_list[i]])#cp.cuda()
                buffer[self.name_agent_list[i]].append(torch.zeros_like(cp).to(self.device_list[self.name_agent_list[i]]))
        return buffer
    
    def clean_buffer(self, buffer):
        for p in buffer:
            p.zero_()
            
    def dist_scalar(self, value, dtype):
        buffer = {}
        for i in range(0, self.world_size):
            buffer[self.name_agent_list[i]] = torch.ones(value, device=self.device_list[self.name_agent_list[i]]).type(dtype)
            
        return buffer
            
    def gossip(self, agent):
        comp_p       = {}
        local_params = {}
    
        for i in range(0, self.world_size):
            #print(self.name_agent_list[i])
            # clone parameters for gossip
            local_params[self.name_agent_list[i]] = []
            for p in agent[self.name_agent_list[i]].policy.parameters():
                cp = p.clone().detach_()
                cp = cp.to(self.device_list[self.name_agent_list[i]])#cp.cuda()
                local_params[self.name_agent_list[i]].append(cp)
            #print('cloning of parameters done')
            # do error compensation step
            for param, error in zip(local_params[self.name_agent_list[i]], self.gossip_error[self.name_agent_list[i]]):
                param.data.add_(error) 
            
            # do the compression
            #print('error compensation is done')
            comp_p[self.name_agent_list[i]] = quantize_layerwise(local_params[self.name_agent_list[i]],
                                                                 self.qlevel, device=self.device_list[self.name_agent_list[i]], is_biased=False)
            # update the error variable
            #print('compression is done')
            for param, comp, error in zip(local_params[self.name_agent_list[i]], comp_p[self.name_agent_list[i]], 
                                            self.gossip_error[self.name_agent_list[i]]):
                error.data.copy_(param)
                error.data.add_(-comp)
        
        print('error compensation and compression step done!')
        local_params = {}
        #update the local models
        #print('averaging the collected gossip')
        for i in range(0, self.world_size):
            self_node  = self.name_agent_list[i]
            neighbours = self.graph_connect[self.name_agent_list[i]]
            self.clean_buffer(self.gossip_var[self_node])
            
            
            # collect the gossip
            for j in range(0, len(neighbours)):
                weight = self.weights[self_node][j] if j>0 else self.weights[self_node][j] - 1.0
                for buffer, gossip in zip(self.gossip_var[self_node], comp_p[neighbours[j]]):
                    gossip.data.mul_(weight.to(self.device_list[neighbours[j]]).type(gossip.dtype))
                    buffer.data.add_(gossip.to(self.device_list[self_node]))
                    
            
            #update params
           
            for params, gossip_buf in zip( agent[self_node].policy.parameters(), self.gossip_var[self_node]):
                gossip_buf.data.mul_(self.averaging_rate[self_node])
                params.data.add_(gossip_buf)
            
            self.clean_buffer(self.gossip_var[self_node])
            
         #clean buffers
        comp_p = {}
        print('gossip average step done!')
            
                
    
            
    
    