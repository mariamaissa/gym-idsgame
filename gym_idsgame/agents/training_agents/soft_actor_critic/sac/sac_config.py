"""
DTO  class holding config parameters for SAC training
"""

import csv

class SACConfig:
    """
    Configuration parameters for SAC
    """

    def __init__(self,
                 input_dim: int,
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
                 hypervec_dim: int = 2048,
                 attacker_output_dim: int = 33,
                 hidden_layer_size: int = 64,
                 replay_memory_size: int = 100000,
                 replay_start_size: int = 10000,
                 batch_size: int = 64,
                 gpu: bool = False,
                 tensorboard: bool = False,
                 tensorboard_dir: str = "",
                 defender_output_dim: int = 33,
                 state_length=1,
                 merged_ad_features: bool = False,
                 normalize_features: bool = False,
                 zero_mean_features: bool = False):
        """
        Initializes the config

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
        :param hypervec_dim: the HDC hypervector size
        :param input_dim: input dimension of the SAC networks
        :param output_dim: output dimension of the SAC networks
        :param attacker_output_dim: output dimensions of the SAC networks for the attacker
        :param defender_output_dim: output dimensions of the SAC networks for the defender
        :param hidden_layer_size: hidden dimension of the SAC networks
        :param num_hidden_layers: number of hidden layers
        :param replay_memory_size: replay memory size
        :param replay_start_size: start size of the replay memory (populated with warmup)
        :param batch_size: the batch size during training
        :param gpu: boolean flag whether using GPU or not
        :param tensorboard: boolean flag whether using tensorboard logging or not
        :param tensorboard_dir: tensorboard logdir
        :param state_length: length of state (Whether stacking observations or not)
        :param merged_ad_features: boolean flag inidicating whether defense and attack features should be merged
        :param normalize_features: boolean flag whether features should be normalized or not
        :param zero_mean_features: boolean flag whether features should be converted to zero-mean vectors
        """
        self.input_dim = input_dim
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
        self.hypervec_dim = hypervec_dim
        self.hidden_layer_size = hidden_layer_size
        self.attacker_output_dim = attacker_output_dim
        self.defender_output_dim = defender_output_dim
        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.gpu = gpu
        self.tensorboard = tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.state_length = state_length
        self.merged_ad_features = merged_ad_features
        self.normalize_features = normalize_features
        self.zero_mean_features = zero_mean_features

    def to_str(self) -> str:
        """
        :return: a string with information about all of the parameters
        """
        return "SAC Hyperparameters: input_dim:{0},policy_lr:{1},critic_lr:{2},alpha_lr:{3}, discount:{4}, tau:{5},"  \
               "alpha_scale:{6},target_update:{7},update_frequency:{8},explore_steps:{9},buffer_size:{10},"  \
               "sample_size:{11},max_steps:{12},hypervec_dim:{13}, attacker_output_dim:{14},hidden_layer_size:{15}," \
               "replay_memory_size:{16},""replay_start_size:{17}, batch_size:{18}, gpu:{19},tensorboard:{20}," \
               "tensorboard_dir:{21},defender_output_dim:{22}, state_length:{23},merged_ad_features:{24}," \
               "normalize_features:{25},zero_mean_features:{26}".format(
            self.input_dim, self.policy_lr, self.critic_lr, self.alpha_lr, self.discount, self.tau,
            self.alpha_scale, self.target_update, self.update_frequency, self.explore_steps, self.buffer_size,
            self.sample_size, self.max_steps, self.hypervec_dim, self.attacker_output_dim, self.hidden_layer_size,
            self.replay_memory_size, self.replay_start_size, self.batch_size, self.gpu, self.tensorboard,
            self.tensorboard_dir, self.defender_output_dim, self.state_length, self.merged_ad_features,
            self.normalize_features, self.zero_mean_features)

    def to_csv(self, file_path: str) -> None:
        """
        Write parameters to csv file

        :param file_path: path to the file
        :return: None
        """
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerow(["input_dim", str(self.input_dim)])
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
            writer.writerow(["hypervec_dim", str(self.hypervec_dim)])
            writer.writerow(["attacker_output_dim", str(self.attacker_output_dim)])
            writer.writerow(["defender_output_dim", str(self.defender_output_dim)])
            writer.writerow(["hidden_layer_size", str(self.hidden_layer_size)])
            writer.writerow(["replay_memory_size", str(self.replay_memory_size)])
            writer.writerow(["replay_start_size", str(self.replay_start_size)])
            writer.writerow(["batch_size", str(self.batch_size)])
            writer.writerow(["gpu", str(self.gpu)])
            writer.writerow(["tensorboard", str(self.tensorboard)])
            writer.writerow(["tensorboard_dir", str(self.tensorboard_dir)])
            writer.writerow(["state_length", str(self.state_length)])
            writer.writerow(["merged_ad_features", str(self.merged_ad_features)])
            writer.writerow(["normalize_features", str(self.normalize_features)])
            writer.writerow(["zero_mean_features", str(self.zero_mean_features)])