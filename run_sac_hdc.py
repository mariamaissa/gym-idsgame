import os
import gymnasium as gym
import time
import sys
import torch
# from gym_idsgame.agents.training_agents.q_learning.abstract_qhd_agent_config import AbstractQHDAgentConfig
from gym_idsgame.agents.training_agents.soft_actor_critic.sac.sac import SACAgent
from gym_idsgame.agents.training_agents.soft_actor_critic.sac.sac_config import SACConfig
from gym_idsgame.agents.training_agents.soft_actor_critic.abstract_sac_agent_config import AbstractSACAgentConfig
from experiments.util import util

NUM_REPEATS = 2

def get_script_path():
    """
    :return: the script path
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = get_script_path()
    return script_dir

# Program entrypoint
    
def main(hdc : bool):
    
    #Just in case
    if torch.cuda.is_available():
        device = f'cuda:{torch.cuda.current_device()}'
    else:
        device = 'cpu'

    _DEVICE = torch.device(device)

    torch.set_default_device(_DEVICE)
    
    scenario = "minimal_defense" #str(sys.argv[1])
    #attacker = True if scenario == "minimal_defense" or scenario == "random_defense" else False

    random_seed = 0
    util.create_artefact_dirs('./', random_seed)

    hparam_dict = {
        'alpha_scale' : [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35],
    }
    
    if hdc:
        type_run = 'hdc'
        hparam = {
        'alpha_lr' : 1e-5,
        'alpha_scale' : .6,
        'critic_lr' : .005,
        'hypervec_dim' : 2048,
        'policy_lr' : 1e-5,
        'sample_size' : 512,
        'tau' : .03
    }
    else:
        type_run = 'nn'
        hparam = {
            'alpha_lr' : 1e-5,
            'alpha_scale' : .7,
            'critic_lr' : .01,
            'policy_lr' : .01,
            'sample_size' : 64,
            'tau' : .005
        }
    
    for key, values in hparam_dict.items():
        for value in values:
            for i in range(NUM_REPEATS):
                
                hparam[key] = value
                
                hdc_sac_config = SACConfig(88,
                                defender_output_dim=88,
                                attacker_output_dim=80,
                                tensorboard=True,
                                tensorboard_dir=f'results/tensorboard/{type_run}/{key}/{value}/{i}',
                                hdc_agent=hdc,
                                **hparam)

                sac_config = AbstractSACAgentConfig(sac_config=hdc_sac_config, num_episodes=10, train_log_frequency=1)

                # Set up environment
                #env_name = "idsgame-" + scenario + "-v3"
                env = gym.make("idsgame-maximal_attack-v3", save_dir=default_output_dir() + "/results/data/" + scenario + "/")
                
                sac_config.to_csv(f'results/logs/{type_run}/{key}/{value}/{i}')

                agent = SACAgent(env, sac_config)
                start = time.time()
                agent.train()
                print("*********Time to train*********: ", time.time() - start)

                #train_result = agent.train_result
               # eval_result = agent.eval_result
               
               
if __name__ == '__main__':
    main(True)
    main(False)