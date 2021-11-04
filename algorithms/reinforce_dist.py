# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 05:51:57 2021

@author: aparna
"""

import numpy as np
import torch
import sys, cv2
import nvidia_smi
from network.agent_batched_pytorch import PedraAgent
from network.communication import communication_setup
from unreal_envs.initial_positions import *
from os import getpid
#from network.Memory import Memory
from aux_functions import *
import os
from util.transformations import euler_from_quaternion
from configs.read_cfg import read_cfg, update_algorithm_cfg
from torchviz import make_dot, make_dot_from_trace



def actorcritic_dist(cfg, env_process, env_folder):
    #torch.cuda.set_device(2)
    algorithm_cfg = read_cfg(config_filename='configs/reinforce_dist.cfg', verbose=True)
    algorithm_cfg.algorithm = cfg.algorithm

    client = []
    # Connect to Unreal Engine and get the drone handle: client
    client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode, num_agents=cfg.num_agents,
                                             client=client)
    initial_pos = old_posit.copy()
    # Load the initial positions for the environment
    reset_array, reset_array_raw, level_name, crash_threshold = initial_positions(cfg.env_name, initZ, cfg.num_agents)
    #print(reset_array, reset_array_raw)
    # Initialize System Handlers
    process = psutil.Process(getpid())

    # Load PyGame Screen
    screen = pygame_connect(phase=cfg.mode)

    fig_z = []
    fig_nav = []
    debug = False
    # Generate path where the weights will be saved
    cfg, algorithm_cfg = save_network_path(cfg=cfg, algorithm_cfg=algorithm_cfg)
    current_state = {}
    new_state = {}
    posit = {}
    device_list = {}
    name_agent_list = []
    data_tuple = {}
    agent = {}
    epi_num = {}
    if cfg.mode == 'train':
        iter = {}
        wait_for_others = {}

        for drone in range(cfg.num_agents):
            name_agent = "drone" + str(drone)
            device_list[name_agent] = torch.device("cuda:{}".format(drone))
            wait_for_others[name_agent] = False
            iter[name_agent] = 1
            epi_num[name_agent] = 1
            data_tuple[name_agent] = []
            name_agent_list.append(name_agent)
            print_orderly(name_agent, 40)
            agent[name_agent] = PedraAgent(algorithm_cfg, client, vehicle_name=name_agent, device=device_list[name_agent])
            current_state[name_agent] = agent[name_agent].get_state()

    elif cfg.mode == 'infer':
        iter = 1
        name_agent = 'drone0'
        name_agent_list.append(name_agent)
        agent[name_agent] = PedraAgent(algorithm_cfg, client, name=name_agent + 'DQN', vehicle_name=name_agent)

        env_cfg = read_cfg(config_filename=env_folder + 'config.cfg')
        nav_x = []
        nav_y = []
        altitude = {}
        altitude[name_agent] = []
        p_z, f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_infer(env_cfg=env_cfg, client=client,
                                                                               env_folder=env_folder)
        nav_text = ax_nav.text(0, 0, '')

        # Select initial position
        reset_to_initial(0, reset_array, client, vehicle_name=name_agent)
        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)

    # Initialize variables

    # iter = 1
    # num_collisions = 0
    episode = {}
    active = True

    print_interval = 1
    automate = True
    choose = False
    print_qval = False
    last_crash = {}
    ret = {}
    distance = {}
    num_collisions = {}
    level = {}
    level_state = {}
    level_posit = {}
    times_switch = {}
    last_crash_array = {}
    ret_array = {}
    distance_array = {}
    epi_env_array = {}
    log_files = {}
    
    #set up communication graph
    comm_setup = communication_setup(agent, name_agent_list, algorithm_cfg, device_list)
    # If the phase is inference force the num_agents to 1
    hyphens = '-' * int((80 - len('Log files')) / 2)
    print(hyphens + ' ' + 'Log files' + ' ' + hyphens)
    for name_agent in name_agent_list:
        ret[name_agent] = 0
        num_collisions[name_agent] = 0
        last_crash[name_agent] = 0
        level[name_agent] = 0
        episode[name_agent] = 0
        level_state[name_agent] = [None] * len(reset_array[name_agent])
        level_posit[name_agent] = [None] * len(reset_array[name_agent])
        times_switch[name_agent] = 0
        last_crash_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        ret_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        distance_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        epi_env_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        distance[name_agent] = 0
        # Log file
        log_path = algorithm_cfg.network_path + name_agent + '\\' + cfg.mode + 'log.txt'
        print("Log path: ", log_path)
        log_files[name_agent] = open(log_path, 'w')

    # switch_env = False

    print_orderly('Simulation begins', 80)

    # save_posit = old_posit

    # last_crash_array = np.zeros(shape=len(level_name), dtype=np.int32)
    # ret_array = np.zeros(shape=len(level_name))
    # distance_array = np.zeros(shape=len(level_name))
    # epi_env_array = np.zeros(shape=len(level_name), dtype=np.int32)

    while active:
        try:
            # active, automate, algorithm_cfg, client = check_user_input(active, automate, agent[name_agent], client,
            #                                                            old_posit[name_agent], initZ, fig_z, fig_nav,
            #                                                            env_folder, cfg, algorithm_cfg)

            if automate:

                if cfg.mode == 'train':

                    if iter[name_agent] % algorithm_cfg.switch_env_steps == 0:
                        switch_env = True
                    else:
                        switch_env = False

                    for name_agent in name_agent_list:
                        while not wait_for_others[name_agent]:
                            start_time = time.time()
                            if switch_env:
                                posit1_old = client.simGetVehiclePose(vehicle_name=name_agent)
                                times_switch[name_agent] = times_switch[name_agent] + 1
                                level_state[name_agent][level[name_agent]] = current_state[name_agent]
                                level_posit[name_agent][level[name_agent]] = posit1_old
                                last_crash_array[name_agent][level[name_agent]] = last_crash[name_agent]
                                ret_array[name_agent][level[name_agent]] = ret[name_agent]
                                distance_array[name_agent][level[name_agent]] = distance[name_agent]
                                epi_env_array[name_agent][level[name_agent]] = episode[name_agent]

                                level[name_agent] = (level[name_agent] + 1) % len(reset_array[name_agent])

                                print(name_agent + ' :Transferring to level: ', level[name_agent], ' - ',
                                      level_name[name_agent][level[name_agent]])

                                if times_switch[name_agent] < len(reset_array[name_agent]):
                                    reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                else:
                                    current_state[name_agent] = level_state[name_agent][level[name_agent]]
                                    posit1_old = level_posit[name_agent][level[name_agent]]
                                    reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                    client.simSetVehiclePose(posit1_old, ignore_collison=True, vehicle_name=name_agent)
                                    time.sleep(0.1)

                                last_crash[name_agent] = last_crash_array[name_agent][level[name_agent]]
                                ret[name_agent] = ret_array[name_agent][level[name_agent]]
                                distance[name_agent] = distance_array[name_agent][level[name_agent]]
                                episode[name_agent] = epi_env_array[name_agent][int(level[name_agent] / 3)]
                                # environ = environ^True
                            else:
                                action = agent[name_agent].choose_action(current_state[name_agent])
                                action_type = 'Prob'
                                # print('Evaluated Policy')
                                action_word = translate_action([action], algorithm_cfg.num_actions)
                                # Take the action
                                # agent[name_agent].take_action(action, algorithm_cfg.num_actions, SimMode=cfg.SimMode)
                                agent[name_agent].take_action([action], algorithm_cfg.num_actions, Mode='static')
                                # print('Took Action')
                                new_state[name_agent] = agent[name_agent].get_state()
                                # print('got state')
                                new_depth1, thresh = agent[name_agent].get_CustomDepth(cfg)
                                # print('Got state')

                                # Get GPS information
                                posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                                position = posit[name_agent].position
                                old_p = np.array(
                                    [old_posit[name_agent].position.x_val, old_posit[name_agent].position.y_val])
                                new_p = np.array([position.x_val, position.y_val])

                                # calculate distance
                                distance[name_agent] = distance[name_agent] + np.linalg.norm(new_p - old_p)
                                old_posit[name_agent] = posit[name_agent]
                                # print('calculated distance')
                                reward, crash = agent[name_agent].reward_gen(new_depth1, action, crash_threshold, thresh,
                                                                             debug, cfg)

                                ret[name_agent] = ret[name_agent] + reward
                                agent[name_agent].store_rewards(reward)
                                agent_state = agent[name_agent].GetAgentState()
                                # print('generated reward')

                                if agent_state.has_collided or distance[name_agent] < 0.01:
                                    num_collisions[name_agent] = num_collisions[name_agent] + 1
                                    if agent_state.has_collided:
                                        print('Crash: Collision detected from environment')
                                    else:
                                        print('Crash: Collision detected from distance')
                                    crash = True
                                    reward = -1

                                data_tuple[name_agent].append([current_state[name_agent], action, reward, crash])

                                # Train if episode ends
                                if crash:
                                    # Don't start another episode unless all agents have ended their episodes
                                    wait_for_others[name_agent] = True
                                    if distance[name_agent] < 0.01:
                                        # Drone won't move, reconnect
                                        print('Recovering from drone mobility issue')

                                        agent[name_agent].client, old_posit, initZ = connect_drone(
                                            ip_address=cfg.ip_address, phase=cfg.mode,
                                            num_agents=cfg.num_agents, client=client)

                                        agent[name_agent].client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address,
                                                                                 phase=cfg.mode,
                                                                                 num_agents=cfg.num_agents,
                                                                                 client=client)
                                        time.sleep(2)
                                        wait_for_others[name_agent] = False


                                    else:
                                        print('Agent:', name_agent, 'Episode: ', epi_num[name_agent], '---- return: ', ret[name_agent],
                                      ' safe flight: ',distance[name_agent], ' episode length: ', len(agent[name_agent].reward_memory))
                                        
                                        # Train episode
                                        
                                        agent[name_agent].learn(data_tuple[name_agent], algorithm_cfg.input_size)
                                        
                                        

                                        data_tuple[name_agent] = []
                                        epi_num[name_agent] += 1
                                        ret[name_agent] = 0
                                        distance[name_agent] = 0
                                        last_crash[name_agent] = 0

                                        # print('train done, now reset')
                                        reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                        # time.sleep(0.2)
                                        current_state[name_agent] = agent[name_agent].get_state()
                                        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)

                                        if epi_num[name_agent] % 1 == 0:
                                            torch.save( agent[name_agent].policy.state_dict(), algorithm_cfg.network_path+'model_'+name_agent+'.h5')

                                        # if all are waiting for others, then we can reset.
                                        if all(wait_for_others.values()):
                                            # Now we can communicate weights
                                            #make_dot(a.mean(), params=dict(agent[name_agent].policy.named_parameters())).render("attached", format="png")  
                                            # for p in agent[name_agent].policy.parameters():
                                            #     print(name_agent, p[0]); break
                                            comm_setup.gossip(agent)
                                           
                                            print('---------------')
                                            for n in name_agent_list:
                                                wait_for_others[n] = False


                                else:
                                    current_state[name_agent] = new_state[name_agent]

                                time_exec = time.time() - start_time
                                gpu_memory, gpu_utilization, sys_memory = get_SystemStats(process, cfg.NVIDIA_GPU)

                                s_log = '{:<6s} - Level {:>2d} - Iter: {:>6d}/{:<5d} {:<8s}-{:>5s} lr: {:>1.8f} Ret = {:>+6.4f} Last Crash = {:<5d} t={:<1.3f} SF = {:<5.4f}  Reward: {:<+1.4f}  '.format(
                                    name_agent,
                                    int(level[name_agent]),
                                    iter[name_agent],
                                    epi_num[name_agent],
                                    action_word,
                                    action_type,
                                    algorithm_cfg.learning_rate,
                                    ret[name_agent],
                                    last_crash[name_agent],
                                    time_exec,
                                    distance[name_agent],
                                    reward)

                                # if iter[name_agent] % print_interval == 0:
                                #     print(s_log)
                                log_files[name_agent].write(s_log + '\n')

                                last_crash[name_agent] = last_crash[name_agent] + 1
                                if debug:
                                    cv2.imshow(name_agent, np.hstack((np.squeeze(current_state[name_agent], axis=0),
                                                                      np.squeeze(new_state[name_agent], axis=0))))
                                    cv2.waitKey(1)

                                if epi_num[name_agent] % algorithm_cfg.total_episodes == 0:
                                    print(automate)
                                    #close_env(env_process)
                                    automate = False

                                iter[name_agent]+=1
                 

                elif cfg.mode == 'infer':
                    # Inference phase
                    agent_state = agent[name_agent].GetAgentState()
                    if agent_state.has_collided:
                        print('Drone collided')
                        print("Total distance traveled: ", np.round(distance[name_agent], 2))
                        active = False
                        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1, vehicle_name=name_agent).join()

                        if nav_x:  # Nav_x is empty if the drone collides in first iteration
                            ax_nav.plot(nav_x.pop(), nav_y.pop(), 'r*', linewidth=20)
                        file_path = env_folder + 'results/'
                        fig_z.savefig(file_path + 'altitude_variation.png', dpi=500)
                        fig_nav.savefig(file_path + 'navigation.png', dpi=500)
                        close_env(env_process)
                        print('Figures saved')
                    else:
                        posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                        distance[name_agent] = distance[name_agent] + np.linalg.norm(np.array(
                            [old_posit[name_agent].position.x_val - posit[name_agent].position.x_val,
                             old_posit[name_agent].position.y_val - posit[name_agent].position.y_val]))
                        # altitude[name_agent].append(-posit[name_agent].position.z_val+p_z)
                        altitude[name_agent].append(-posit[name_agent].position.z_val - f_z)

                        quat = (posit[name_agent].orientation.w_val, posit[name_agent].orientation.x_val,
                                posit[name_agent].orientation.y_val, posit[name_agent].orientation.z_val)
                        yaw = euler_from_quaternion(quat)[2]

                        x_val = posit[name_agent].position.x_val
                        y_val = posit[name_agent].position.y_val
                        z_val = posit[name_agent].position.z_val

                        nav_x.append(env_cfg.alpha * x_val + env_cfg.o_x)
                        nav_y.append(env_cfg.alpha * y_val + env_cfg.o_y)
                        nav.set_data(nav_x, nav_y)
                        nav_text.remove()
                        nav_text = ax_nav.text(25, 55, 'Distance: ' + str(np.round(distance[name_agent], 2)),
                                               style='italic',
                                               bbox={'facecolor': 'white', 'alpha': 0.5})

                        line_z.set_data(np.arange(len(altitude[name_agent])), altitude[name_agent])
                        ax_z.set_xlim(0, len(altitude[name_agent]))
                        fig_z.canvas.draw()
                        fig_z.canvas.flush_events()

                        current_state[name_agent] = agent[name_agent].get_state()
                        action = agent[name_agent].choose_action(current_state[name_agent])
                        action_type = 'Prob'
                        action_word = translate_action([action], algorithm_cfg.num_actions)
                        # Take continuous action
                        # agent[name_agent].take_action(action, algorithm_cfg.num_actions, Mode='static')

                        agent[name_agent].take_action([action], algorithm_cfg.num_actions, Mode='static')
                        old_posit[name_agent] = posit[name_agent]

                        # Verbose and log making
                        s_log = 'Position = ({:<3.2f},{:<3.2f}, {:<3.2f}) Orientation={:<1.3f} Predicted Action: {:<8s}  '.format(
                            x_val, y_val, z_val, yaw, action_word
                        )

                        print(s_log)
                        log_files[name_agent].write(s_log + '\n')



        except Exception as e:
            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')

                client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                                         num_agents=cfg.num_agents, client=client)
                time.sleep(2)
                agent[name_agent].client = client
                wait_for_others[name_agent] = False
            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                print('Hit r and then backspace to start from this point')