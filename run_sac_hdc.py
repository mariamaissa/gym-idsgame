import os
import gym
import time
import sys
# from gym_idsgame.agents.training_agents.q_learning.abstract_qhd_agent_config import AbstractQHDAgentConfig
from gym_idsgame.agents.training_agents.soft_actor_critic.sac.sac import SACAgent
from gym_idsgame.agents.training_agents.soft_actor_critic.sac.sac_config import SACConfig
from gym_idsgame.agents.training_agents.soft_actor_critic.abstract_sac_agent_config import AbstractSACAgentConfig
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
        hdc_sac_config = SACConfig(33)

        qhd_agent_config = AbstractSACAgentConfig(sac_config=hdc_sac_config)

        # Set up environment
        env_name = "idsgame-" + scenario + "-v3"
        env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + scenario + "/")

        agent = SACConfig(env, qhd_agent_config)
        start = time.time()
        agent.train()
        print("*********Time to train*********: ", time.time() - start)

        train_result = agent.train_result
        eval_result = agent.eval_result