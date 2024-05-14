"""
Configuration for AbstractSACAgent
"""
import csv
import os
from gym_idsgame.agents.training_agents.soft_actor_critic.sac.sac_config import SACConfig
from torch.utils.tensorboard import SummaryWriter


class AbstractSACAgentConfig:
    """
    DTO with configuration for SACAgent
    """

    def __init__(self,
                 render: bool = False,
                 eval_sleep: float = 0.35,
                 eval_episodes: int = 1,
                 train_log_frequency: int = 100,
                 eval_log_frequency: int = 1,
                 video: bool = False,
                 video_fps: int = 5,
                 video_dir: bool = None,
                 num_episodes: int = 5000,
                 eval_render: bool = False,
                 gifs: bool = False,
                 gif_dir: str = None,
                 eval_frequency: int = 1000,
                 video_frequency: int = 101,
                 attacker: bool = True,
                 defender: bool = False,
                 save_dir: str = None,
                 attacker_load_path: str = None,
                 defender_load_path: str = None,
                 sac_config: SACConfig = None,  # Changed from  DQNConfig
                 checkpoint_freq: int = 100000,
                 random_seed: int = 0,
                 tab_full_state_space: bool = False):
        """
        Initialize environment and hyperparameters

        :param render: whether to render the environment *during training*
        :param eval_sleep: amount of sleep between time-steps during evaluation and rendering
        :param eval_episodes: number of evaluation episodes
        :param train_log_frequency: number of episodes between logs during train
        :param eval_log_frequency: number of episodes between logs during eval
        :param video: boolean flag whether to record video of the evaluation.
        :param video_dir: path where to save videos (will overwrite)
        :param gif_dir: path where to save gifs (will overwrite)
        :param num_episodes: number of training epochs
        :param eval_render: whether to render the game during evaluation or not
                            (perhaps set to False if video is recorded instead)
        :param gifs: boolean flag whether to save gifs during evaluation or not
        :param eval_frequency: the frequency (episodes) when running evaluation
        :param video_frequency: the frequency (eval episodes) to record video and gif
        :param attacker: True if the QAgent is an attacker
        :param attacker: True if the QAgent is a defender
        :param save_dir: dir to save Q-table
        :param attacker_load_path: path to load a saved Q-table of the attacker
        :param defender_load_path: path to load a saved Q-table of the defender
        :param sac_config: configuration for SAC
        :param checkpoint_freq: frequency of checkpointing the model (episodes)
        :param random_seed: the random seed for reproducibility
        :param tab_full_state_space: a boolean flag indicating whether the tabular q learning approach use full
                                     state space or not
        """
        self.render = render
        self.eval_sleep = eval_sleep
        self.eval_episodes = eval_episodes
        self.train_log_frequency = train_log_frequency
        self.eval_log_frequency = eval_log_frequency
        self.video = video
        self.video_fps = video_fps
        self.video_dir = video_dir
        self.num_episodes = num_episodes
        self.eval_render = eval_render
        self.gifs = gifs
        self.gif_dir = gif_dir
        self.eval_frequency = eval_frequency
        self.logger = None
        self.video_frequency = video_frequency
        self.attacker = attacker
        self.defender = defender
        self.save_dir = save_dir
        self.attacker_load_path = attacker_load_path
        self.defender_load_path = defender_load_path
        self.sac_config = sac_config
        self.checkpoint_freq = checkpoint_freq
        self.random_seed = random_seed
        self.tab_full_state_space = tab_full_state_space

    def to_str(self) -> str:
        """
        :return: a string with information about all of the parameters
        """
        return "Hyperparameters: render:{0},eval_sleep:{1},eval_episodes:{2},train_log_frequency:{3}," \
               "eval_log_frequency:{4},video:{5},video_fps:{6},video_dir:{7},num_episodes:{8},eval_render:{9}," \
               "gifs:{10},gifdir:{11},eval_frequency:{12},video_frequency:{13},attacker{14},defender:{15}," \
               "checkpoint_freq:{16},random_seed:{17},tab_full_state_space:{18}".format(
             self.render, self.eval_sleep,  self.eval_episodes, self.train_log_frequency,
            self.eval_log_frequency, self.video, self.video_fps, self.video_dir, self.num_episodes, self.eval_render,
            self.gifs, self.gif_dir, self.eval_frequency, self.video_frequency, self.attacker, self.defender,
            self.checkpoint_freq, self.random_seed, self.tab_full_state_space)

    def to_csv(self, file_path: str) -> None:
        """
        Write parameters to csv file

        :param file_path: path to the file
        :return: None
        """
        
        os.makedirs(file_path, exist_ok=True)
        
        with open(f'{file_path}/log.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerow(["render", str(self.render)])
            writer.writerow(["eval_sleep", str(self.eval_sleep)])
            writer.writerow(["eval_episodes", str(self.eval_episodes)])
            writer.writerow(["train_log_frequency", str(self.train_log_frequency)])
            writer.writerow(["eval_log_frequency", str(self.eval_log_frequency)])
            writer.writerow(["video", str(self.video)])
            writer.writerow(["video_fps", str(self.video_fps)])
            writer.writerow(["video_dir", str(self.video_dir)])
            writer.writerow(["num_episodes", str(self.num_episodes)])
            writer.writerow(["eval_render", str(self.eval_render)])
            writer.writerow(["gifs", str(self.gifs)])
            writer.writerow(["gifdir", str(self.gif_dir)])
            writer.writerow(["eval_frequency", str(self.eval_frequency)])
            writer.writerow(["video_frequency", str(self.video_frequency)])
            writer.writerow(["attacker", str(self.attacker)])
            writer.writerow(["defender", str(self.defender)])
            writer.writerow(["checkpoint_freq", str(self.checkpoint_freq)])
            writer.writerow(["random_seed", str(self.random_seed)])
            writer.writerow(["tab_full_state_space", str(self.tab_full_state_space)])
            if self.sac_config is not None:
                writer.writerow(["input_dim", str(self.sac_config.input_dim)])
                writer.writerow(["policy_lr", str(self.sac_config.policy_lr)])
                writer.writerow(["critic_lr", str(self.sac_config.critic_lr)])
                writer.writerow(["alpha_lr", str(self.sac_config.alpha_lr)])
                writer.writerow(["discount", str(self.sac_config.discount)])
                writer.writerow(["tau", str(self.sac_config.tau)])
                writer.writerow(["alpha_scale", str(self.sac_config.alpha_scale)])
                writer.writerow(["target_update", str(self.sac_config.target_update)])
                writer.writerow(["update_frequency", str(self.sac_config.update_frequency)])
                writer.writerow(["buffer_size", str(self.sac_config.buffer_size)])
                writer.writerow(["sample_size", str(self.sac_config.sample_size)])
                writer.writerow(["max_steps", str(self.sac_config.max_steps)])
                writer.writerow(["hypervec_dim", str(self.sac_config.hypervec_dim)])
                writer.writerow(["output_dim", str(self.sac_config.attacker_output_dim)])
                writer.writerow(["hidden_layer_size", str(self.sac_config.hidden_layer_size)])
                writer.writerow(["replay_memory_size", str(self.sac_config.replay_memory_size)])
                writer.writerow(["replay_start_size", str(self.sac_config.replay_start_size)])
                writer.writerow(["batch_size", str(self.sac_config.batch_size)])
                writer.writerow(["gpu", str(self.sac_config.gpu)])
                writer.writerow(["tensorboard", str(self.sac_config.tensorboard)])
                writer.writerow(["tensorboard_dir", str(self.sac_config.tensorboard_dir)])

    def hparams_dict(self):
        hparams = {}
        hparams["eval_episodes"] = self.eval_episodes
        hparams["train_log_frequency"] = self.train_log_frequency
        hparams["eval_log_frequency"] = self.eval_log_frequency
        hparams["num_episodes"] = self.num_episodes
        hparams["eval_frequency"] = self.eval_frequency
        hparams["attacker"] = self.attacker
        hparams["defender"] = self.defender
        hparams["checkpoint_freq"] = self.checkpoint_freq
        hparams["random_seed"] = self.random_seed
        hparams["tab_full_state_space"] = self.tab_full_state_space
        if self.sac_config is not None:
            hparams["input_dim"] = self.sac_config.input_dim
            hparams["policy_lr"] = self.sac_config.policy_lr
            hparams["critic_lr"] = self.sac_config.critic_lr
            hparams["alpha_lr"] = self.sac_config.alpha_lr
            hparams["discount"] = self.sac_config.discount
            hparams["tau"] = self.sac_config.tau
            hparams["alpha_scale"] = self.sac_config.alpha_scale
            hparams["target_update"] = self.sac_config.target_update
            hparams["update_frequency"] = self.sac_config.update_frequency
            hparams["buffer_size"] = self.sac_config.buffer_size
            hparams["sample_size"] = self.sac_config.sample_size
            hparams["max_steps"] = self.sac_config.max_steps
            hparams["hdc_agent"] = self.sac_config.hdc_agent
            hparams["hypervec_dim"] = self.sac_config.hypervec_dim
            hparams["output_dim"] = self.sac_config.attacker_output_dim
            hparams["hidden_layer_size"] = self.sac_config.hidden_layer_size
            hparams["replay_memory_size"] = self.sac_config.replay_memory_size
            hparams["replay_start_size"] = self.sac_config.replay_start_size
            hparams["batch_size"] = self.sac_config.batch_size
            hparams["gpu"] = self.sac_config.gpu
        return hparams
