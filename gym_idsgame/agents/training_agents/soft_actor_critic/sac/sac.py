"""
An agent for the IDSGameEnv that implements the SAC algorithm.
"""
from copy import deepcopy
from typing import Union
import numpy as np
import time
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.agents.training_agents.soft_actor_critic.abstract_sac_agent_config import AbstractSACAgentConfig
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.envs.constants import constants
from gym_idsgame.agents.training_agents.models.fnn_w_linear import FNNwithLinear
from gym_idsgame.agents.training_agents.soft_actor_critic.abstract_sac_agent import AbstractSACAgent

from .implemented_agents import NNAgent, HDCAgent, Transition, MemoryBuffer

class SACAgent(AbstractSACAgent):
    def __init__(self, env: IdsGameEnv, config: AbstractSACAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(SACAgent, self).__init__(env, config)
        self.attacker = None
        self.defender = None
        self.tensorboard_writer = SummaryWriter(self.config.sac_config.tensorboard_dir)
        self.attacker_buffer = MemoryBuffer(config.sac_config.replay_memory_size, config.sac_config.sample_size)
        self.defender_buffer = deepcopy(self.attacker_buffer) #In case of future buffer where they need to share exact attributes
        
        #We figure this out here so we can use device when we are making transitions
        if torch.cuda.is_available() and self.config.sac_config.gpu:
            self.device = torch.device("cuda:0")
            self.config.logger.info("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            self.config.logger.info("Running on the CPU")
            
        self.initialize_models()
        self.tensorboard_writer.add_hparams(self.config.hparams_dict(), {})
        self.env.idsgame_config.save_trajectories = False
        self.env.idsgame_config.save_attack_stats = False
        
        

    def warmup(self) -> None:
        """
        A warmup without any learning just to populate the replay buffer following a random strategy

        :return: None
        """

        # Setup logging
        outer_warmup = tqdm.tqdm(total=self.config.sac_config.replay_start_size, desc='Warmup', position=0)
        outer_warmup.set_description_str("[Warmup] step:{}, buffer_size: {}".format(0, 0))

        # Reset env
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs
        obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
        obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
        obs = (obs_state_a, obs_state_d)
        self.config.logger.info("Starting warmup phase to fill replay buffer")

        # Perform <self.config.sac_config.replay_start_size> steps and fill the replay memory
        for i in range(self.config.sac_config.replay_start_size):

            if i % self.config.train_log_frequency == 0:
                log_str = "[Warmup] step:{}, buffer_size: {}".format(i, self.attacker_buffer.size())
                outer_warmup.set_description_str(log_str)
                self.config.logger.info(log_str)

            # Select random attacker and defender actions
            attacker_actions = list(range(self.env.num_attack_actions))
            defender_actions = list(range(self.env.num_defense_actions))
            legal_attack_actions = list(filter(lambda action: self.env.is_attack_legal(action), attacker_actions))
            legal_defense_actions = list(filter(lambda action: self.env.is_defense_legal(action), defender_actions))
            attacker_action = torch.tensor(np.random.choice(legal_attack_actions), device=self.device, dtype=torch.float32)
            defender_action = torch.tensor(np.random.choice(legal_defense_actions), device=self.device, dtype=torch.float32)
            action = (attacker_action, defender_action)

            # Take action in the environment
            obs_prime, reward, done, info = self.env.step(action)
            attacker_obs_prime, defender_obs_prime = obs_prime
            obs_state_a_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=True, state=[])
            obs_state_d_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=False, state=[])
            obs_prime = (obs_state_a_prime, obs_state_d_prime)

            # Add transition to replay memory
            a_trans, d_trans = self._env_to_transition(obs, action, reward, done, obs_prime)
            self.attacker_buffer.add_data(a_trans)
            self.defender_buffer.add_data(d_trans)

            # Move to new state
            obs = obs_prime
            outer_warmup.update(1)

            if done:
                obs = self.env.reset(update_stats=False)
                attacker_obs, defender_obs = obs
                obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
                obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
                obs = (obs_state_a, obs_state_d)

        self.config.logger.info("{} Warmup steps completed, replay buffer size: {}".format(
            self.config.sac_config.replay_start_size, self.attacker_buffer.size()))
        self.env.close()

        try:
            # Add network graph to tensorboard with a sample batch as input

            if self.config.attacker:
                s_1 = self.attacker_buffer.sample().state
                self.tensorboard_writer.add_graph(self.attacker, s_1)

            if self.config.defender:
                s_1 = self.defender_buffer.sample().state
                self.tensorboard_writer.add_graph(self.defender, s_1)
        except:
            self.config.logger.warning("Error when trying to add network graph to tensorboard")

    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """
        
        input_dict = {
                'input_size' : self.config.sac_config.input_dim,
                'output_size' : self.config.sac_config.attacker_output_dim,
                'policy_lr' : self.config.sac_config.policy_lr,
                'critic_lr' : self.config.sac_config.critic_lr,
                'alpha_lr' : self.config.sac_config.alpha_lr,
                'discount' : self.config.sac_config.discount,
                'tau' : self.config.sac_config.tau,
                'alpha_scale' : self.config.sac_config.alpha_scale,
                'target_update' : self.config.sac_config.target_update,
                'update_frequency' : self.config.sac_config.update_frequency,
                'summary_writer' : self.tensorboard_writer,
                'extra_info' : 'attacker'
            }
        
        if self.config.sac_config.hdc_agent:
            input_dict['hyper_dim'] = self.config.sac_config.hypervec_dim
            self.attacker = HDCAgent(**input_dict)
            input_dict['output_size'] = self.config.sac_config.defender_output_dim
            input_dict['extra_info'] = 'defender'
            self.defender = HDCAgent(**input_dict)
            
        else:
            self.attacker = NNAgent(**input_dict)
            input_dict['output_size'] = self.config.sac_config.defender_output_dim
            input_dict['extra_info'] = 'defender'
            self.defender = NNAgent(**input_dict)


        self.attacker.to(self.device)
        self.defender.to(self.device)

    def get_action(self, state: np.ndarray, eval : bool = False, attacker : bool = True) -> int:
        """

        :param state: the state to sample an action for
        :param eval: boolean flag whether running in evaluation mode
        :param attacker: boolean flag whether running in attacker mode (if false assume defender)
        :return: The sampled action id
        """
        state = torch.from_numpy(state.flatten()).float()

        # Move to GPU if using GPU
        if torch.cuda.is_available() and self.config.sac_config.gpu:
            device = torch.device("cuda:0")
            state = state.to(device)

        if attacker:
            actions = list(range(self.env.num_attack_actions))
            legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))
        else:
            actions = list(range(self.env.num_defense_actions))
            legal_actions = list(filter(lambda action: self.env.is_defense_legal(action), actions))

        with torch.no_grad():
            if attacker:
                act_values = torch.zeros(self.config.sac_config.attacker_output_dim).index_fill(0, self.attacker(state), 1)
            else:
                act_values = torch.zeros(self.config.sac_config.defender_output_dim).index_fill(0, self.attacker(state), 1)

        return legal_actions[torch.argmax(act_values[legal_actions]).item()]

    def train(self) -> ExperimentResult:
        """
        Runs the SAC algorithm

        :return: Experiment result
        """
        self.config.logger.info("Starting Warmup")
        self.warmup()
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        obs = self.env.reset(update_stats=False)[0]
        attacker_obs, defender_obs = obs
        obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
        obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
        obs = (obs_state_a, obs_state_d)
        attacker_obs, defender_obs = obs

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []
        episode_avg_attacker_loss = []
        episode_avg_defender_loss = []

        # Logging
        self.outer_train.set_description_str("[Train]avg_a_R:{:.2f},avg_d_R:{:.2f},"
                                             "avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                                             "acc_D_R:{:.2f}".format(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Training
        for episode in range(self.config.num_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.render:
                    self.env.render(mode="human")

                if not self.config.attacker and not self.config.defender:
                    raise AssertionError("Must specify whether training an attacker agent or defender agent")

                # Default initialization
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_action = self.get_action(attacker_obs, attacker=True)
                if self.config.defender:
                    defender_action = self.get_action(defender_obs, attacker=False)

                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _, _ = self.env.step(action)
                attacker_obs_prime, defender_obs_prime = obs_prime
                obs_state_a_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=True, state=[])
                obs_state_d_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=False, state=[])
                obs_prime = (obs_state_a_prime, obs_state_d_prime)

                # Add transition to replay memory
                a_tran, d_tran = self._env_to_transition(obs, action, reward, done, obs_prime)
                self.attacker_buffer.add_data(a_tran)
                self.defender_buffer.add_data(d_tran)


                # Perform a gradient descent step of the Q-network using targets produced by target network
                if self.config.attacker:
                    self.attacker.update(self.attacker_buffer.sample())

                if self.config.defender:
                    self.defender.update(self.defender_buffer.sample())


                # Update metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1

                # Move to the next state
                obs = obs_prime
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender

            # Render final frame
            if self.config.render:
                self.env.render(mode="human")


            # Record episode metrics
            self.num_train_games += 1
            self.num_train_games_total += 1
            if self.env.state.hacked:
                self.num_train_hacks += 1
                self.num_train_hacks_total += 1
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            
            """
            I think my implementation should take care of the losses
            if episode_step > 0:
                if self.config.attacker:
                    episode_avg_attacker_loss.append(episode_attacker_loss/episode_step)
                if self.config.defender:
                    episode_avg_defender_loss.append(episode_defender_loss / episode_step)
            else:
                if self.config.attacker:
                    episode_avg_attacker_loss.append(episode_attacker_loss)
                if self.config.defender:
                    episode_avg_defender_loss.append(episode_defender_loss)
            """

            episode_steps.append(episode_step)

            # Log average metrics every <self.config.train_log_frequency> episodes
            if episode % self.config.train_log_frequency == 0:
                if self.num_train_games > 0 and self.num_train_games_total > 0:
                    self.train_hack_probability = self.num_train_hacks / self.num_train_games
                    self.train_cumulative_hack_probability = self.num_train_hacks_total / self.num_train_games_total
                else:
                    self.train_hack_probability = 0.0
                    self.train_cumulative_hack_probability = 0.0
                self.log_metrics(episode, self.train_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 episode_avg_attacker_loss, episode_avg_defender_loss)

                # Log values and gradients of the parameters (histogram summary) to tensorboard

                """ #Leaving this here because good idea to track grads in personal implementation
                if self.config.attacker:
                    for tag, value in self.attacker.named_parameters():
                        tag = tag.replace('.', '/')
                        self.tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), episode)
                        self.tensorboard_writer.add_histogram(tag + '_attacker/grad', value.grad.data.cpu().numpy(),
                                                              episode)

                if self.config.defender:
                    for tag, value in self.defender.named_parameters():
                        tag = tag.replace('.', '/')
                        self.tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), episode)
                        self.tensorboard_writer.add_histogram(tag + '_defender/grad', value.grad.data.cpu().numpy(),
                                                              episode)
                """

                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_steps = []
                self.num_train_games = 0
                self.num_train_hacks = 0


            # Run evaluation every <self.config.eval_frequency> episodes
            #if episode % self.config.eval_frequency == 0:
            #    self.eval(episode)

            # Save models every <self.config.checkpoint_frequency> episodes
            """ #TODO Do we want this?
            if episode % self.config.checkpoint_freq == 0:
                self.save_model()
                self.env.save_trajectories(checkpoint=True)
                self.env.save_attack_data(checkpoint=True)
                if self.config.save_dir is not None:
                    time_str = str(time.time())
                    self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
                    self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")
            """

            # Reset environment for the next episode and update game stats
            done = False
            obs = self.env.reset(update_stats=True)[0]
            attacker_obs, defender_obs = obs
            obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
            obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
            obs = (obs_state_a, obs_state_d)
            attacker_obs, defender_obs = obs
            self.outer_train.update(1)

        self.config.logger.info("Training Complete")

        # Final evaluation (for saving Gifs etc)
        #self.eval(self.config.num_episodes-1, log=False)

        # Save Q-networks
        self.save_model()

        # Save other game data
        self.env.save_trajectories(checkpoint=False)
        self.env.save_attack_data(checkpoint=False)
        if self.config.save_dir is not None:
            time_str = str(time.time())
            self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
            self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

        return self.train_result


    # TODO: @Ian this function may be unnecessary
    def eval(self, train_episode, log=True) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :param train_episode: the train episode to keep track of logging
        :param log: whether to log the result
        :return: None
        """
        raise ValueError('I am not supposed to be here')
        self.config.logger.info("Starting Evaluation")
        time_str = str(time.time())

        self.num_eval_games = 0
        self.num_eval_hacks = 0

        if len(self.eval_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting eval with non-empty result object")
        if self.config.eval_episodes < 1:
            return
        done = False

        # Video config
        if self.config.video:
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
                                      video_frequency=self.config.video_frequency)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_eval = tqdm.tqdm(total=self.config.eval_episodes, desc='Eval Episode', position=1)
        self.outer_eval.set_description_str(
            "[Eval] avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
            "acc_D_R:{:.2f}".format(0.0, 0,0, 0.0, 0.0, 0.0, 0.0))

        # Eval
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs
        obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
        obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
        obs = (obs_state_a, obs_state_d)
        attacker_obs, defender_obs = obs

        for episode in range(self.config.eval_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)

                # Default initialization
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_action = self.get_action(attacker_obs, eval=True, attacker=True)
                if self.config.defender:
                    defender_action = self.get_action(defender_obs, eval=True, attacker=False)
                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _ = self.env.step(action)
                attacker_obs_prime, defender_obs_prime = obs_prime
                obs_state_a_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=True, state=[])
                obs_state_d_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=False, state=[])
                obs_prime = (obs_state_a_prime, obs_state_d_prime)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender

            # Render final frame when game completed
            if self.config.eval_render:
                self.env.render()
                time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, episode_step))

            # Record episode metrics
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Update eval stats
            self.num_eval_games +=1
            self.num_eval_games_total += 1
            if self.env.state.detected:
                self.eval_attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
            if self.env.state.hacked:
                self.eval_attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.num_eval_hacks += 1
                self.num_eval_hacks_total +=1

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0 and log:
                if self.num_eval_hacks > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                if self.num_eval_games_total > 0:
                    self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                        self.num_eval_games_total)
                self.log_metrics(episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 eval = True, update_stats=False)

            # Save gifs
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(train_episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

                # Add frames to tensorboard
                for idx, frame in enumerate(self.env.episode_frames):
                    self.tensorboard_writer.add_image(str(train_episode) + "_eval_frames/" + str(idx),
                                                       frame, global_step=train_episode,
                                                      dataformats = "HWC")


            # Reset for new eval episode
            done = False
            obs = self.env.reset(update_stats=False)
            attacker_obs, defender_obs = obs
            obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
            obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
            obs = (obs_state_a, obs_state_d)
            attacker_obs, defender_obs = obs
            self.outer_eval.update(1)

        # Log average eval statistics
        if log:
            if self.num_eval_hacks > 0:
                self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
            if self.num_eval_games_total > 0:
                self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                    self.num_eval_games_total)

            self.log_metrics(train_episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                             episode_steps, eval=True, update_stats=True)

        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def save_model(self) -> None:
        """
        Saves the PyTorch Model Weights

        :return: None
        """
        time_str = str(time.time())
        if self.config.save_dir is not None:
            if self.config.attacker:
                path = self.config.save_dir + "/" + time_str + "_attacker_q_network.pt"
                self.config.logger.info("Saving SAC model to: {}".format(path))
                self.attacker.save_actor(extra_info=path)
            if self.config.defender:
                path = self.config.save_dir + "/" + time_str + "_defender_q_network.pt"
                self.config.logger.info("Saving Q-network to: {}".format(path))
                self.defender.save_actor(extra_info=path)
        else:
            self.config.logger.warning("Save path not defined, not saving SAC model to disk")


    def update_state(self, attacker_obs: np.ndarray = None, defender_obs: np.ndarray = None,
                     state: np.ndarray = None, attacker: bool = True) -> np.ndarray:
        """
        Update approximative Markov state

        :param attacker_obs: attacker obs
        :param defender_obs: defender observation
        :param state: current state
        :param attacker: boolean flag whether it is attacker or not
        :return: new state
        """
        if attacker and self.env.idsgame_config.game_config.reconnaissance_actions:
            #if not self.env.local_view_features():
            a_obs_len = self.env.idsgame_config.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:, a_obs_len:a_obs_len+self.env.idsgame_config.game_config.num_attack_types]
            if self.env.idsgame_config.reconnaissance_bool_features:
                d_bool_features = attacker_obs[:, a_obs_len+self.env.idsgame_config.game_config.num_attack_types:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]
            # else:
            #     a_obs_len = self.env.idsgame_config.game_config.num_attack_types + 1
            #     defender_obs = attacker_obs[:,
            #                    a_obs_len:a_obs_len + self.env.idsgame_config.game_config.num_attack_types]
            #     if self.env.idsgame_config.reconnaissance_bool_features:
            #         d_bool_features = attacker_obs[:,
            #                           a_obs_len + self.env.idsgame_config.game_config.num_attack_types:]
            #     attacker_obs = attacker_obs[:, 0:a_obs_len]

        if not attacker and self.env.local_view_features():
            attacker_obs = self.env.state.get_attacker_observation(
                self.env.idsgame_config.game_config.network_config,
                local_view=False,
                reconnaissance=self.env.idsgame_config.reconnaissance_actions)


        # Zero mean
        if self.config.sac_config.zero_mean_features:
            if not self.env.local_view_features() or not attacker:
                attacker_obs_1 = attacker_obs[:, 0:-1]
            else:
                attacker_obs_1 = attacker_obs[:, 0:-2]
            zero_mean_attacker_features = []
            for idx, row in enumerate(attacker_obs_1):
                mean = np.mean(row)
                if mean != 0:
                    t = row - mean
                else:
                    t = row
                if np.isnan(t).any():
                    t = attacker_obs[idx]
                else:
                    t = t.tolist()
                    if not self.env.local_view_features() or not attacker:
                        t.append(attacker_obs[idx][-1])
                    else:
                        t.append(attacker_obs[idx][-2])
                        t.append(attacker_obs[idx][-1])
                zero_mean_attacker_features.append(t)

            defender_obs_1 = defender_obs[:, 0:-1]
            zero_mean_defender_features = []
            for idx, row in enumerate(defender_obs_1):
                mean = np.mean(row)
                if mean != 0:
                    t = row - mean
                else:
                    t = row
                if np.isnan(t).any():
                    t = defender_obs[idx]
                else:
                    t = t.tolist()
                    t.append(defender_obs[idx][-1])
                zero_mean_defender_features.append(t)

            attacker_obs = np.array(zero_mean_attacker_features)
            defender_obs = np.array(zero_mean_defender_features)

        # Normalize
        if self.config.sac_config.normalize_features:
            if not self.env.local_view_features() or not attacker:
                attacker_obs_1 = attacker_obs[:, 0:-1] / np.linalg.norm(attacker_obs[:, 0:-1])
            else:
                attacker_obs_1 = attacker_obs[:, 0:-2] / np.linalg.norm(attacker_obs[:, 0:-2])
            normalized_attacker_features = []
            for idx, row in enumerate(attacker_obs_1):
                if np.isnan(attacker_obs_1).any():
                    t = attacker_obs[idx]
                else:
                    t = row.tolist()
                    if not self.env.local_view_features() or not attacker:
                        t.append(attacker_obs[idx][-1])
                    else:
                        t.append(attacker_obs[idx][-2])
                        t.append(attacker_obs[idx][-1])
                normalized_attacker_features.append(t)

            if attacker and self.env.idsgame_config.game_config.reconnaissance_actions:
                defender_obs_1 = defender_obs[:, 0:-1] / np.linalg.norm(defender_obs[:, 0:-1])
            else:
                defender_obs_1 = defender_obs / np.linalg.norm(defender_obs)
            normalized_defender_features = []
            for idx, row in enumerate(defender_obs_1):
                if np.isnan(defender_obs_1).any():
                    t = defender_obs[idx]
                else:
                    if attacker and self.env.idsgame_config.game_config.reconnaissance_actions:
                        t = row.tolist()
                        t.append(defender_obs[idx][-1])
                    else:
                        t = row

                normalized_defender_features.append(t)

            attacker_obs = np.array(normalized_attacker_features)
            defender_obs = np.array(normalized_defender_features)

        if self.env.local_view_features() and attacker:
            if not self.env.idsgame_config.game_config.reconnaissance_actions:
                neighbor_defense_attributes = np.zeros((attacker_obs.shape[0], defender_obs.shape[1]))
                for node in range(attacker_obs.shape[0]):
                    id = int(attacker_obs[node][-1])
                    neighbor_defense_attributes[node] = defender_obs[id]
            else:
                neighbor_defense_attributes = defender_obs

        if self.env.fully_observed() or \
                (self.env.idsgame_config.game_config.reconnaissance_actions and attacker):
            if self.config.sac_config.merged_ad_features:
                if not self.env.local_view_features() or not attacker:
                    a_pos = attacker_obs[:, -1]
                    if not self.env.idsgame_config.game_config.reconnaissance_actions:
                        det_values = defender_obs[:, -1]
                        temp = defender_obs[:, 0:-1] - attacker_obs[:, 0:-1]
                    else:
                        temp = defender_obs[:, 0:] - attacker_obs[:, 0:-1]
                    features = []
                    for idx, row in enumerate(temp):
                        t = row.tolist()
                        t.append(a_pos[idx])
                        if self.env.fully_observed():
                            t.append(det_values[idx])
                        features.append(t)
                else:
                    node_ids = attacker_obs[:, -1]
                    if not self.env.idsgame_config.game_config.reconnaissance_actions:
                        det_values = neighbor_defense_attributes[:, -1]
                    if not self.env.idsgame_config.game_config.reconnaissance_actions:
                        temp = neighbor_defense_attributes[:, 0:-1] - attacker_obs[:, 0:-1]
                    else:
                        temp = np.full(neighbor_defense_attributes.shape, -1)
                        for i in range(len(neighbor_defense_attributes)):
                            if np.sum(neighbor_defense_attributes[i]) > 0:
                                temp[i] = neighbor_defense_attributes[i] - attacker_obs[i, 0:-1]
                    features = []
                    for idx, row in enumerate(temp):
                        t = row.tolist()
                        t.append(node_ids[idx])
                        #t.append(node_reachable[idx])
                        if not self.env.idsgame_config.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                features = np.array(features)
                if self.env.idsgame_config.reconnaissance_bool_features:
                    f = np.zeros((features.shape[0], features.shape[1] + d_bool_features.shape[1]))
                    for i in range(features.shape[0]):
                        f[i] = np.append(features[i], d_bool_features[i])
                    features = f
                if self.config.sac_config.state_length == 1:
                    return features
                if len(state) == 0:
                    s = np.array([features] * self.config.sac_config.state_length)
                    return s
                state = np.append(state[1:], np.array([features]), axis=0)
                return state
            else:
                if not self.env.local_view_features() or not attacker:
                    if self.env.idsgame_config.game_config.reconnaissance_actions and attacker:
                        combined_features = []
                        for idx, row in enumerate(attacker_obs):
                            combined_row = np.append(row, defender_obs[idx])
                            combined_features.append(combined_row)
                        if self.env.idsgame_config.reconnaissance_bool_features:
                            combined_features = np.array(combined_features)
                            f = np.zeros(
                                (combined_features.shape[0], combined_features.shape[1] + d_bool_features.shape[1]))
                            for i in range(combined_features.shape[0]):
                                f[i] = np.append(combined_features[i], d_bool_features[i])
                            combined_features = f
                        return np.array(combined_features)

                    return np.append(attacker_obs, defender_obs)
                else:
                    if self.env.idsgame_config.reconnaissance_bool_features:
                        f = np.zeros((attacker_obs.shape[0],
                                      attacker_obs.shape[1] + neighbor_defense_attributes.shape[1] +
                                      d_bool_features.shape[1]))
                        for i in range(f.shape[0]):
                            f[i] = np.append(np.append(attacker_obs[i], neighbor_defense_attributes[i]),
                                             d_bool_features[i])
                    else:
                        f = np.zeros((attacker_obs.shape[0],
                                      attacker_obs.shape[1] + neighbor_defense_attributes.shape[1]))
                        for i in range(f.shape[0]):
                            f[i] = np.append(attacker_obs[i], neighbor_defense_attributes[i])
                if self.config.sac_config.state_length == 1:
                    return f
                if len(state) == 0:
                    s = np.array([f] * self.config.sac_config.state_length)
                    return s
                # if not self.env.local_view_features() or not attacker:
                #     temp = np.append(attacker_obs, defender_obs)
                # else:
                #     temp = np.append(attacker_obs, neighbor_defense_attributes)
                state = np.append(state[1:], np.array([f]), axis=0)
            return state
        else:
            if self.config.sac_config.state_length == 1:
                if attacker:
                    return np.array(attacker_obs)
                else:
                    return np.array(defender_obs)
            if len(state) == 0:
                if attacker:
                    return np.array([attacker_obs] * self.config.sac_config.state_length)
                else:
                    return np.array([defender_obs] * self.config.sac_config.state_length)
            if attacker:
                state = np.append(state[1:], np.array([attacker_obs]), axis=0)
            else:
                state = np.append(state[1:], np.array([defender_obs]), axis=0)
            return state

    def _env_to_transition(self,
                           state : tuple[np.ndarray, np.ndarray],
                           actions : tuple[torch.Tensor, torch.Tensor], #They will be scalars so need to add dimension
                           reward : tuple[float, float],
                           done : bool,
                           next_state : tuple[np.ndarray, np.ndarray]) -> tuple[Transition, Transition]:
        """
        Will take in inputs from the environment and create a transition for the
        attacker and the defender
        """
        
        a_s, d_s = state
        a_a, d_a = actions
        a_r, d_r = reward
        a_ns, d_ns = next_state
        
        attacker_transition = Transition(
            torch.tensor(a_s.flatten(), device=self.device, dtype=torch.float32),
            torch.tensor([a_a], device=self.device, dtype=torch.int64),
            torch.tensor(a_ns.flatten(), device=self.device, dtype=torch.float32),
            torch.tensor([a_r], device=self.device, dtype=torch.float32),
            torch.tensor([done], device=self.device, dtype=torch.float32)
        )
        
        defender_transition = Transition(
            torch.tensor(d_s.flatten(), device=self.device, dtype=torch.float32),
            torch.tensor([d_a], device=self.device, dtype=torch.int64),
            torch.tensor(d_ns.flatten(), device=self.device, dtype=torch.float32),
            torch.tensor([d_r], device=self.device, dtype=torch.float32),
            torch.tensor([done], device=self.device, dtype=torch.float32)
        )
        
        return attacker_transition, defender_transition
        