import os
import gym
import time
import sys
# from gym_idsgame.agents.training_agents.q_learning.abstract_qhd_agent_config import AbstractQHDAgentConfig
from gym_idsgame.agents.training_agents.soft_actor_critic.hdc_sac.hdc_sac import SACAgent
from gym_idsgame.agents.training_agents.soft_actor_critic.hdc_sac.hdc_sac_config import SACConfig
from experiments.util import util

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
if __name__ == '__main__':
    scenario = str(sys.argv[1])
    attacker = True if scenario == "minimal_defense" or scenario == "random_defense" else False

    random_seed = 0
    util.create_artefact_dirs('./', random_seed)

    for lr in [0.00001]:
        sac_config = SACConfig(input_dim=88,
                               defender_output_dim=88,  # attacker would need 80: 10 attacks (+1 for defender), 8 nodes
                               attacker_output_dim=80,
                               replay_memory_size=10000,
                               batch_size=32,
                               target_network_update_freq=1000,  # TODO: Hyperparameter for fine-tuning
                               gpu=False,
                               tensorboard=False,
                               tensorboard_dir=default_output_dir() + "/results/tensorboard/",
                               lr_exp_decay=False,
                               lr_decay_rate=0.9999)
#AbstractQHDAgentConfig
        sac_agent_config = SACAgentConfig(gamma=0.999,
                                      lr=0.00001,  # TODO: Hyper-parameter for fine-tuning
                                      num_episodes=20001,
                                      epsilon=1,
                                      min_epsilon=0.01,
                                      epsilon_decay=0.95, # TODO: Hyperparameter for fine-tuning
                                      eval_sleep=0.9,
                                      eval_frequency=1000,
                                      eval_episodes=100,
                                      train_log_frequency=100,
                                      eval_log_frequency=1,
                                      render=False,
                                      eval_render=False,
                                      video=False,
                                      video_fps=5,
                                      video_frequency=101,
                                      video_dir=default_output_dir() + "/results/videos/",
                                      gifs=False,
                                      gif_dir=default_output_dir() + "/results/gifs/",
                                      save_dir=default_output_dir() + "/results/data/" + scenario + "/",
                                      attacker=attacker,
                                      defender=not attacker,
                                      qhd_config=qhd_config,
                                      checkpoint_freq=300000)

        # Set up environment
        env_name = "idsgame-" + scenario + "-v3"
        env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + scenario + "/")

        agent = SACAgent(env, sac_agent_config, "")
        start = time.time()
        agent.train()
        print("*********Time to train*********: ", time.time() - start)

        train_result = agent.train_result
        eval_result = agent.eval_result