import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from gym_idsgame.agents.training_agents.soft_actor_critic.sac.implemented_agents.data_collection import Transition
from .hdc_implementation import TargetQFunction, QFunction, Actor, Alpha
from .encoders import RBFEncoder, EXPEncoder

class Agent:

    def __init__(self, 
                 input_size : int,
                 output_size : int,
                 hyper_dim : int,
                 policy_lr : float,
                 critic_lr : float,
                 alpha_lr : float,
                 discount : float,
                 tau : float,
                 alpha_scale : float,
                 target_update : int, #When the target should update
                 update_frequency : int, #When the models should update,
                 summary_writer : SummaryWriter,
                 extra_info : str = ''
                 ) -> None:
        
        self._actor_encoder = RBFEncoder(input_size, hyper_dim)
        self._critic_encoder = EXPEncoder(input_size, hyper_dim)

        self._target_q = TargetQFunction(tau, None)
        self._alpha = Alpha(output_size, alpha_scale, alpha_lr, extra_info)

        self._actor = Actor(hyper_dim,
                            output_size,
                            policy_lr,
                            self._actor_encoder,
                            self._critic_encoder,
                            self._alpha,
                            self._target_q,
                            extra_info)
        
        self._q_function = QFunction(hyper_dim,
                                     output_size,
                                     self._actor_encoder,
                                     self._critic_encoder,
                                     self._actor,
                                     self._target_q,
                                     self._alpha,
                                     critic_lr,
                                     discount,
                                     extra_info)
        
        self._target_q.set_actual(self._q_function)
        
        self._target_update = target_update
        self._update_freq = update_frequency

        self._sw = summary_writer
        
        self._steps = 1

    def __call__(self, state : Tensor) -> Tensor:
        """Will return the action that should be executed at the given state"""
        with torch.no_grad():
            hvec_state = self._actor_encoder(state)
            action, _, _ = self._actor(hvec_state)
            return action

    def update(self, batch : Transition) -> None:
        """Will update the networks according to the correct steps"""
        if self._steps % self._update_freq == 0:
            ce_state = self._q_function.update(batch, self._steps, self._sw)
            self._actor.update(batch, self._steps, self._sw, ce_state)

        if self._steps % self._target_update == 0:
            self._target_q.update()
            
        self._steps += 1

    def save_actor(self, extra_info : str = '') -> None:
        """Will save the actor to a file named bestweights_extrainfo"""
        self._actor.save(f'bestweights_{extra_info}.pt')

    def to(self, device) -> None:
        """Moves agents assets to the device"""
        self._q_function.to(device)
        self._actor.to(device)
        self._target_q.to(device)
        self._alpha.to(device)