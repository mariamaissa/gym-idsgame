import os
import gym
import time
import sys
from gym_idsgame.agents.training_agents.soft_actor_critic.abstract_sac_agent_config import AbstractSACAgentConfig
from gym_idsgame.agents.training_agents.soft_actor_critic.sac.sac import SACAgent
from gym_idsgame.agents.training_agents.soft_actor_critic.sac.sac_config import SACConfig
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
    scenario = str(sys.argv[1]) # TODO: @Ian: you can either hardcode the scenario or input it
    # @Ian: The scenarios are: `minimal_defense`, `random_defense`, `maximal_attack` `random_attack`
    attacker = True if scenario == "minimal_defense" or scenario == "random_defense" else False
    random_seed = 0
    util.create_artefact_dirs('./', random_seed)


    sac_config = SACConfig(input_dim=88,
    # TODO: @Ian: you can uncomment parameters if you want to fine-tune
                           # policy_lr= 3e-4,
                           # critic_lr = 3e-4,
                           # alpha_lr= 3e-4,
                           # discount = .99,
                           # tau = .005,
                           # alpha_scale = .89,
                           # target_update = 1,
                           # update_frequency = 1,
                           # explore_steps = 0,
                           # buffer_size = 10 ** 6,
                           # sample_size = 64,
                           # max_steps = 1e5,
                           # hypervec_dim = 2048,
                           defender_output_dim=88,
                           attacker_output_dim=80,
                           replay_memory_size=10000,
                           batch_size=32,
                           gpu=False,
                           tensorboard=False,
                           tensorboard_dir=default_output_dir() + "/results/tensorboard/",
                           )
#AbstractQHDAgentConfig
    sac_agent_config = AbstractSACAgentConfig(
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
                                  sac_config=sac_config,
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