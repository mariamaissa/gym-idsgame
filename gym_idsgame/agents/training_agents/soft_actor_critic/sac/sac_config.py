"""
DTO  class holding config parameters for SAC training
"""

import csv

class SACConfig:
    """
    Configuration parameters for SAC
    """

    def __init__(self,
                 input_dim: int,  # TODO @Ian: do we need this?
                 output_dim: int,  # TODO @Ian: do we need this?
                 attacker_output_dim: int = 33,
                 hidden_layer_size: int = 64,
                 replay_memory_size: int = 100000,
                 replay_start_size: int = 10000,
                 batch_size: int = 64,
                 num_hidden_layers=2,           # TODO @Ian: do we need this?
                 # target_network_update_freq: int = 10, # TODO @Ian:do you want to inclde this in your model interface?
                 gpu: bool = False,
                 tensorboard: bool = False,
                 tensorboard_dir: str = "",
                 # loss_fn: str = "MSE",        # TODO: @Ian delete?
                 # optimizer: str = "Adam",     # TODO: @Ian delete?
                 # lr_exp_decay,                # TODO: @Ian delete?
                 # lr_decay_rate,               # TODO: @Ian delete?
                 # hidden_activation: str = "ReLU",  # TODO: @Ian delete?
                 defender_output_dim: int = 33,
                 state_length=1,
                 merged_ad_features: bool = False,
                 normalize_features: bool = False,
                 zero_mean_features: bool = False):
        """
        Initializes the config

        :param input_dim: input dimension of the SAC networks
        :param output_dim: output dimension of the SAC networks
        :param attacker_output_dim: output dimensions of the SAC networks for the attacker
        :param defender_output_dim: output dimensions of the SAC networks for the defender
        :param hidden_layer_size: hidden dimension of the SAC networks
        :param num_hidden_layers: number of hidden layers
        :param replay_memory_size: replay memory size
        :param replay_start_size: start size of the replay memory (populated with warmup)
        :param batch_size: the batch size during training
        :param target_network_update_freq: the frequency (in episodes) of updating the target network   # TODO: @Mariam delete
        :param gpu: boolean flag whether using GPU or not
        :param tensorboard: boolean flag whether using tensorboard logging or not
        :param tensorboard_dir: tensorboard logdir
        :param loss_fn: loss function   # TODO: @Mariam delete
        :param optimizer: optimizer     # TODO: @Mariam delete
        :param hidden_activation: the activation function for hidden units  # TODO: @Mariam delete
        :param state_length: length of state (Whether stacking observations or not)
        :param merged_ad_features: boolean flag inidicating whether defense and attack features should be merged
        :param normalize_features: boolean flag whether features should be normalized or not
        :param zero_mean_features: boolean flag whether features should be converted to zero-mean vectors
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_size = hidden_layer_size
        self.attacker_output_dim = attacker_output_dim
        self.defender_output_dim = defender_output_dim
        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        # self.target_network_update_freq = target_network_update_freq
        self.gpu = gpu
        self.tensorboard = tensorboard
        self.tensorboard_dir = tensorboard_dir
        # self.loss_fn = loss_fn
        # self.optimizer = optimizer
        self.num_hidden_layers = num_hidden_layers
        # self.lr_exp_decay = lr_exp_decay
        # self.lr_decay_rate = lr_decay_rate
        # self.hidden_activation = hidden_activation
        self.state_length = state_length
        self.merged_ad_features = merged_ad_features
        self.normalize_features = normalize_features
        self.zero_mean_features = zero_mean_features

    def to_str(self) -> str:
        """
        :return: a string with information about all of the parameters
        """
        #TODO: @Mariam Delete unnecessary params
        return "SAC Hyperparameters: input_dim:{0},attacker_output_dim:{1},hidden_layer_size:{2},replay_memory_size:{3}," \
               "replay_start_size:{4}," \
               "batch_size:{5},target_network_update_freq:{6},gpu:{7},tensorboard:{8}," \
               "tensorboard_dir:{9},loss_fn:{10},optimizer:{11},num_hidden_layers:{12}," \
               "lr_exp_decay:{13},lr_decay_rate:{14},hidden_activation:{15},defender_output_dim:{16}," \
               "state_length:{17},merged_ad_features:{18},normalize_features:{19},zero_mean_features:{20}".format(
            self.input_dim,
            self.attacker_output_dim,
            self.hidden_layer_size,
            self.replay_memory_size,
            self.replay_start_size,
            self.batch_size,
            None, #self.target_network_update_freq,
            self.gpu,
            self.tensorboard,
            self.tensorboard_dir,
            None, #self.loss_fn,
            None, #self.optimizer,
            self.num_hidden_layers,
            None, #self.lr_exp_decay,
            None, #self.lr_decay_rate,
            None, #self.hidden_activation,
            self.defender_output_dim,
            self.state_length,
            self.merged_ad_features,
            self.normalize_features,
            self.zero_mean_features)

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
            writer.writerow(["attacker_output_dim", str(self.attacker_output_dim)])
            writer.writerow(["defender_output_dim", str(self.defender_output_dim)])
            writer.writerow(["hidden_layer_size", str(self.hidden_layer_size)])
            writer.writerow(["replay_memory_size", str(self.replay_memory_size)])
            writer.writerow(["replay_start_size", str(self.replay_start_size)])
            writer.writerow(["batch_size", str(self.batch_size)])
            # writer.writerow(["target_network_update_freq", str(self.target_network_update_freq)])
            writer.writerow(["gpu", str(self.gpu)])
            writer.writerow(["tensorboard", str(self.tensorboard)])
            writer.writerow(["tensorboard_dir", str(self.tensorboard_dir)])
            # writer.writerow(["loss_fn", str(self.loss_fn)])
            # writer.writerow(["optimizer", str(self.optimizer)])
            writer.writerow(["num_hidden_layers", str(self.num_hidden_layers)])
            # writer.writerow(["lr_exp_decay", str(self.lr_exp_decay)])
            # writer.writerow(["lr_decay_rate", str(self.lr_decay_rate)])
            # writer.writerow(["hidden_activation", str(self.hidden_activation)])
            writer.writerow(["state_length", str(self.state_length)])
            writer.writerow(["merged_ad_features", str(self.merged_ad_features)])
            writer.writerow(["normalize_features", str(self.normalize_features)])
            writer.writerow(["zero_mean_features", str(self.zero_mean_features)])