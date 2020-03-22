"""
RL environment for an intrusion detection Markov game
"""
from typing import Union
import numpy as np
import gym
import os
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.random_defense_agent import RandomDefenseBotAgent
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
import gym_idsgame.envs.util.idsgame_util as util
from gym_idsgame.envs.constants import constants

class IdsGameEnv(gym.Env):
    """
    Implementation of the RL environment from the paper
    "Adversarial Reinforcement Learning in a Cyber Security Simulation" by Edlerman et. al.

    It is an abstract very simple cyber security simulation where there is an attacker agent that tries to penetrate
    a network and a defender agent that tries to defend the network.
    """

    def __init__(self, idsgame_config: IdsGameConfig = None):
        """
        Initializes the environment

        Observation:
            Type: Box(num_nodes*num_attack_types)
        Actions:
            Type: Discrete(num_nodes*num_action_types)
        Reward:
            Reward is 0 for all steps except the final step which is either +100 (win) or -100 (loss)
        Starting State:
            Start node, all attack values are 0
        Episode Termination:
            When attacker reaches DATA node or when attacker is detected

        :param idsgame_config: configuration of the environment
        """
        if idsgame_config is None:
            idsgame_config = IdsGameConfig()
        util.validate_config(idsgame_config)
        self.idsgame_config: IdsGameConfig = idsgame_config
        self.state: GameState = self.idsgame_config.game_config.initial_state.copy()
        self.observation_space = self.idsgame_config.game_config.get_attacker_observation_space()
        self.defender = self.idsgame_config.defender_agent
        self.action_space = self.idsgame_config.game_config.get_attacker_action_space()
        self.viewer = None
        self.steps_beyond_done = None
        self.metadata = {
         'render.modes': ['human', 'rgb_array'],
         'video.frames_per_second' : 50 # Video rendering speed
        }
        self.reward_range = (float(constants.GAME_CONFIG.NEGATIVE_REWARD), float(constants.GAME_CONFIG.POSITIVE_REWARD))
        self.num_states = self.idsgame_config.game_config.num_nodes
        self.num_actions = self.idsgame_config.game_config.num_actions

    # -------- API ------------

    def step(self, action) -> Union[np.ndarray, int, bool, dict]:
        """
        Takes a step in the environment using the given action.


        When end of episode is reached, the caller is responsible for calling `reset()`
        to reset this environment's state.

        :param action: the action to take in the environment
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Initialization
        reward = 0
        info = {}
        self.state.attack_events = []
        self.state.defense_events = []

        if self.state.game_step > constants.GAME_CONFIG.MAX_GAME_STEPS:
            return self.state.get_attacker_observation(self.idsgame_config.game_config.network_config), 0, True, info

        # 1. Interpret attacker action
        attacker_pos = self.state.attacker_pos
        target_node_id, target_pos, attack_type = util.interpret_attack(action, self.idsgame_config.game_config)
        #print("target node: {}".format(target_node_id))
        if target_node_id == self.idsgame_config.game_config.num_nodes-1:
            print("target start node")

        # 2. Sample defense action
        defense_pos, defense_type, defense_node_id = self.__defense_action()

        # 3. Defend
        self.state.defend(defense_node_id, defense_type, self.idsgame_config.game_config.max_value,
                          self.idsgame_config.game_config.network_config)
        self.state.add_defense_event(defense_pos, defense_type)

        if util.is_attack_legal(attacker_pos, target_pos, self.idsgame_config.game_config.num_cols,
                                self.idsgame_config.game_config.network_config.adjacency_matrix):
            # 4. Attack
            self.state.attack(target_node_id, attack_type, self.idsgame_config.game_config.max_value,
                              self.idsgame_config.game_config.network_config)
            self.state.add_attack_event(target_pos, attack_type)

            # 5. Simulate attack outcome
            attack_successful = self.state.simulate_attack(target_node_id, attack_type,
                                                           self.idsgame_config.game_config.network_config)

            # 6. Update state based on attack outcome
            if attack_successful:
                self.state.attacker_pos = target_pos
                if target_pos == self.idsgame_config.game_config.network_config.data_pos:
                    self.state.done = True
                    self.state.hacked = True
                    reward = constants.GAME_CONFIG.POSITIVE_REWARD
                else:
                    detected = self.state.simulate_detection(target_node_id)
                    if detected:
                        self.state.done = True
                        self.state.detected = True
                        reward = -constants.GAME_CONFIG.POSITIVE_REWARD
        if self.state.done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                gym.logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                                "You should always call 'reset()' once you receive 'done = True' -- "
                                "any further steps are undefined behavior.")
                self.steps_beyond_done += 1
        self.state.game_step += 1
        observation = self.state.get_attacker_observation(self.idsgame_config.game_config.network_config)
        if self.viewer is not None:
            self.viewer.gameframe.set_state(self.state)
        return observation, reward, self.state.done, info

    def reset(self) -> None:
        """
        Resets the environment and returns the initial state

        :return: the initial state
        """
        self.steps_beyond_done = None
        self.state.new_game(self.idsgame_config.game_config.initial_state)
        if self.viewer is not None:
            self.viewer.gameframe.reset()
        observation = self.state.get_attacker_observation(self.idsgame_config.game_config.network_config)
        return observation

    def restart(self) -> None:
        """
        Restarts the game, and all the history

        :return: the observation from the first state
        """
        obs = self.reset()
        self.state.restart()
        return obs

    def render(self, mode: str ='human'):
        """
        Renders the environment

        Supported rendering modes:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.

        :param mode: the rendering mode
        :return: True (if human mode) otherwise an rgb array
        """
        if mode not in self.metadata["render.modes"]:
            raise NotImplemented("mode: {} is not supported".format(mode))
        if self.viewer is None:
            self.__setup_viewer()
        arr = self.viewer.render(return_rgb_array = mode=='rgb_array')
        self.state.attack_events = []
        self.state.defense_events = []
        return arr

    def close(self) -> None:
        """
        Closes the viewer (cleanup)

        :return: None
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_attacker_node_from_observation(self, observation: np.ndarray) -> int:
        """
        Extracts which node the attacker is currently at from the observation representation

        :param observation: the observation representation emitted from the environment
        :return: the id of the node that the attacker is in
        """
        return self.state.get_attacker_node_from_observation(observation)

    def hack_probabiltiy(self) -> float:
        """
        :return: the hack-probabiltiy according to the game history
        """
        hack_probability = 0.0
        if self.state.num_hacks > 0:
            hack_probability = float(self.state.num_hacks) / float(self.state.num_games)
        return hack_probability

    def is_attack_legal(self, attack_action:int) -> bool:
        """
        Check if a given attack is legal or not.

        :param attack_action: the attack to verify
        :return: True if legal otherwise False
        """
        return util.is_attack_id_legal(attack_action, self.idsgame_config.game_config, self.state.attacker_pos)

    # -------- Private methods ------------

    def __setup_viewer(self):
        """
        Setup for the viewer to use for rendering
        :return: None
        """
        from gym_idsgame.envs.rendering.viewer import Viewer
        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, './rendering/', constants.RENDERING.RESOURCES_DIR)
        self.idsgame_config.render_config.resources_dir = resource_path
        self.viewer = Viewer(idsgame_config=self.idsgame_config)
        self.viewer.agent_start()

    def __defense_action(self) -> Union[Union[int, int], int, int]:
        """
        Utility method that samples an action from a defender strategy

        :return: position of the node to defend, defense-type, defense-node-id
        """
        defense_row, defense_col, defense_type = self.defender.action(self.state)
        defense_pos = (defense_row, defense_col)
        defense_node_id = self.idsgame_config.game_config.network_config.get_node_id(defense_pos)
        return defense_pos, defense_type, defense_node_id

# -------- Concrete envs ------------
class IdsGameRandomDefense1L1S10ADEnv(IdsGameEnv):
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        defender_policy = RandomDefenseBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_policy)
        super().__init__(idsgame_config=idsgame_config)