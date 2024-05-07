"""
Configuration for AbstractSACAgent
"""
import csv
from gym_idsgame.agents.training_agents.soft_actor_critic.sac.sac_config import SACConfig
from torch.utils.tensorboard import SummaryWriter


class AbstractSACAgentConfig:
    """
    DTO with configuration for SACAgent
    """

    def __init__(self,
                 policy_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 discount: float = .99,
                 tau: float = .005,
                 alpha_scale: float = .89,
                 target_update: int = 1,  # When the target should update
                 update_frequency: int = 1,  # When the models should update,
                 explore_steps: int = 0,
                 buffer_size: int = 10 ** 6,
                 sample_size: int = 64,
                 max_steps: int = 1e5,
                 hdc_agent: bool = False,  # TODO: @Ian added this to differentiate between models (but change if needed)
                 hypervec_dim: int = 2048,
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

        :param hidden_size: specifies the hidden layer size
        :param policy_lr: Policy (Actor) learning rate
        :param critic_lr: Critic learning rate
        :param alpha_lr: Temperature learning rate
        :param discount: 
        :param tau:
        :param alpha_scale:
        :param target_update:
        :param update_frequency:
        :param explore_steps:
        :param buffer_size:
        :param sample_size:
        :param max_steps:
        :param hdc_agent: determine which model to use
        :param hypervec_dim: the HDC hypervector size
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
        self.hidden_size = hidden_size
        self.policy_lr = policy_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.discount = discount
        self.tau = tau
        self.alpha_scale = alpha_scale
        self.target_update = target_update
        self.update_frequency = update_frequency
        self.explore_steps = explore_steps
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.max_steps = max_steps
        self.hdc_agent = hdc_agent
        self.hypervec_dim = hypervec_dim
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
        return "Hyperparameters: hidden_size:{0},policy_lr:{1},critic_lr:{2},alpha_lr:{3}, discount:{4}, tau:{5},"  \
               "alpha_scale:{6},target_update:{7},update_frequency:{8},explore_steps:{9},buffer_size:{10},"  \
               "sample_size:{11},max_steps:{12},hdc_agent:{13},hypervec_dim:{14},render:{15},eval_sleep:{16},"  \
               "eval_episodes:{17},train_log_frequency:{18},eval_log_frequency:{19},video:{20},video_fps:{21},"  \
               "video_dir:{22},num_episodes:{23},eval_render:{24},gifs:{25},gifdir:{26},eval_frequency:{27},"  \
               "video_frequency:{28},attacker{29},defender:{30},checkpoint_freq:{31},random_seed:{32},"  \
               "tab_full_state_space:{33}".format(
            self.hidden_size, self.policy_lr, self.critic_lr, self.alpha_lr, self.discount, self.tau,
            self.alpha_scale, self.target_update, self.update_frequency, self.explore_steps, self.buffer_size,
            self.sample_size, self.max_steps, self.hdc_agent, self.hypervec_dim, self.render, self.eval_sleep,
            self.eval_episodes, self.train_log_frequency, self.eval_log_frequency, self.video, self.video_fps,
            self.video_dir, self.num_episodes, self.eval_render, self.gifs, self.gif_dir, self.eval_frequency,
            self.video_frequency, self.attacker, self.defender, self.checkpoint_freq, self.random_seed,
            self.tab_full_state_space)

    def to_csv(self, file_path: str) -> None:
        """
        Write parameters to csv file

        :param file_path: path to the file
        :return: None
        """
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerow(["hidden_size", str(self.hidden_size)])
            writer.writerow(["policy_lr", str(self.policy_lr)])
            writer.writerow(["critic_lr", str(self.critic_lr)])
            writer.writerow(["alpha_lr", str(self.alpha_lr)])
            writer.writerow(["discount", str(self.discount)])
            writer.writerow(["tau", str(self.tau)])
            writer.writerow(["alpha_scale", str(self.alpha_scale)])
            writer.writerow(["target_update", str(self.target_update)])
            writer.writerow(["update_frequency", str(self.update_frequency)])
            writer.writerow(["explore_steps", str(self.explore_steps)])
            writer.writerow(["buffer_size", str(self.buffer_size)])
            writer.writerow(["sample_size", str(self.sample_size)])
            writer.writerow(["max_steps", str(self.max_steps)])
            writer.writerow(["hdc_agent", str(self.hdc_agent)])
            writer.writerow(["hypervec_dim", str(self.hypervec_dim)])
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
            # TODO: @Mariam: review what Ian says with SACConfig comments
            if self.sac_config is not None:
                writer.writerow(["input_dim", str(self.sac_config.input_dim)])
                writer.writerow(["output_dim", str(self.sac_config.attacker_output_dim)])
                writer.writerow(["hidden_layer_size", str(self.sac_config.hidden_layer_size)])
                writer.writerow(["replay_memory_size", str(self.sac_config.replay_memory_size)])
                writer.writerow(["replay_start_size", str(self.sac_config.replay_start_size)])
                writer.writerow(["batch_size", str(self.sac_config.batch_size)])
                # writer.writerow(["target_network_update_freq", str(self.sac_config.target_network_update_freq)])
                writer.writerow(["gpu", str(self.sac_config.gpu)])
                writer.writerow(["tensorboard", str(self.sac_config.tensorboard)])
                writer.writerow(["tensorboard_dir", str(self.sac_config.tensorboard_dir)])
                # writer.writerow(["loss_fn", str(self.sac_config.loss_fn)])
                # writer.writerow(["optimizer", str(self.sac_config.optimizer)])
                writer.writerow(["num_hidden_layers", str(self.sac_config.num_hidden_layers)])
                # writer.writerow(["lr_exp_decay", str(self.sac_config.lr_exp_decay)])
                # writer.writerow(["lr_decay_rate", str(self.sac_config.lr_decay_rate)])
                # writer.writerow(["hidden_activation", str(self.sac_config.hidden_activation)])

    def hparams_dict(self):
        hparams = {}
        hparams["hidden_size"] = self.hidden_size
        hparams["policy_lr"] = self.policy_lr
        hparams["critic_lr"] = self.critic_lr
        hparams["alpha_lr"] = self.alpha_lr
        hparams["discount"] = self.discount
        hparams["tau"] = self.tau
        hparams["alpha_scale"] = self.alpha_scale
        hparams["target_update"] = self.target_update
        hparams["update_frequency"] = self.update_frequency
        hparams["buffer_size"] = self.buffer_size
        hparams["sample_size"] = self.sample_size
        hparams["max_steps"] = self.max_steps
        hparams["hdc_agent"] = self.hdc_agent
        hparams["hypervec_dim"] = self.hypervec_dim
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
        # TODO: @Mariam: review what Ian says with SACConfig comments
        if self.sac_config is not None:
            hparams["input_dim"] = self.sac_config.input_dim
            hparams["output_dim"] = self.sac_config.attacker_output_dim
            hparams["hidden_layer_size"] = self.sac_config.hidden_layer_size
            hparams["replay_memory_size"] = self.sac_config.replay_memory_size
            hparams["replay_start_size"] = self.sac_config.replay_start_size
            hparams["batch_size"] = self.sac_config.batch_size
            hparams["num_hidden_layers"] = self.sac_config.num_hidden_layers
            # hparams["target_network_update_freq"] = self.sac_config.target_network_update_freq
            hparams["gpu"] = self.sac_config.gpu
            # hparams["loss_fn"] = self.sac_config.loss_fn
            hparams["optimizer"] = self.sac_config.optimizer
            # hparams["lr_exp_decay"] = self.sac_config.lr_exp_decay
            # hparams["lr_decay_rate"] = self.sac_config.lr_decay_rate
            # hparams["hidden_activation"] = self.sac_config.hidden_activation
        return hparams
