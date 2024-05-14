from torch import Tensor
import torch
from torch.utils.tensorboard import SummaryWriter

from gym_idsgame.agents.training_agents.soft_actor_critic.sac.implemented_agents.data_collection import Transition
from .nn_implementation import QFunctionTarget, QFunction, Actor, Alpha

HIDDEN_SIZE = 256 #We can easily change this later if we wanted to

class Agent:

    def __init__(self, 
                 input_size : int,
                 output_size : int,
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
        
        self._target_q = QFunctionTarget(None, tau)
        self._alpha = Alpha(output_size, alpha_scale, alpha_lr, extra_info)

        self._actor = Actor(input_size, 
                            output_size, 
                            HIDDEN_SIZE,
                            self._target_q,
                            self._alpha, 
                            policy_lr,
                            extra_info)
        
        self._q_function = QFunction(input_size,
                                     output_size,
                                     HIDDEN_SIZE,
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
            action, _, _ = self._actor(state)
            return action

    def update(self, batch : Transition) -> None:
        """Will update the networks according to the correct steps"""
        if self._steps % self._update_freq == 0:
            self._q_function.update(batch, self._steps, self._sw)
            self._actor.update(batch, self._steps, self._sw)

        if self._steps % self._target_update == 0:
            self._target_q.update()
            
        self._steps += 1

    def save_actor(self, extra_info : str = '') -> None:
        """Will save the actor to file named bestweights with extra_info concatenated to the end"""
        self._actor.save(f'bestweights_{extra_info}.pt')

    def to(self, device) -> None:
        """Moves agents assets to the device"""
        self._q_function.to(device)
        self._actor.to(device)
        self._target_q.to(device)
        self._alpha.to(device)