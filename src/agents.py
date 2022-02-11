import abc
import random

import torch
from Box2D import b2World
from dataclasses import dataclass

from src.soft_body import SoftBody
from src.utils import create_soft_body


class BaseController(abc.ABC):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __str__(self):
        return "BaseController[input={},output={}]".format(self.input_dim, self.output_dim)

    @abc.abstractmethod
    def control(self, obs):
        pass

    @classmethod
    def create_controller(cls, input_dim, output_dim, solution):
        if not solution:
            return RandomController(input_dim, output_dim)
        return MLPController(input_dim, output_dim, solution)


class RandomController(BaseController):

    def __init__(self, input_dim, output_dim):
        super(RandomController, self).__init__(input_dim, output_dim)

    def __str__(self):
        return super(RandomController, self).__str__().replace("Base", "Random")

    def control(self, obs):
        return [random.random() * 2.0 - 1.0 for _ in range(self.output_dim)]


class TrivialController(BaseController):

    def __init__(self, input_dim, output_dim):
        super(TrivialController, self).__init__(input_dim, output_dim)
        self.prev_action = -1

    def __str__(self):
        return super(TrivialController, self).__str__().replace("Base", "Trivial")

    def control(self, obs):
        self.prev_action = -1 if self.prev_action == 1 else 1
        return [-1 if self.prev_action == 1 else 1 for _ in range(self.output_dim)]


class MLPController(BaseController):

    def __init__(self, input_dim, output_dim, solution):
        super(MLPController, self).__init__(input_dim, output_dim)
        self.nn = torch.nn.Sequential(torch.nn.Linear(in_features=self.input_dim, out_features=self.input_dim),
                                      torch.nn.Tanh(),
                                      torch.nn.Linear(in_features=self.input_dim, out_features=self.output_dim),
                                      torch.nn.Tanh()
                                      )
        self.set_params(solution)
        for param in self.nn.parameters():
            param.requires_grad = False

    def __str__(self):
        return super(MLPController, self).__str__().replace("Base", "MLP")

    def set_params(self, params):
        state_dict = self.nn.state_dict()
        start = 0
        for key, coeffs in state_dict.items():
            num = coeffs.numel()
            state_dict[key] = torch.tensor(params[start:start + num])
            start += num

    def control(self, obs):
        return self.nn(torch.from_numpy(obs).float()).detach().numpy()


@dataclass
class Agent(object):
    morphology: SoftBody
    controller: BaseController
    world: b2World

    def act(self):
        obs = self.morphology.sense()
        control = self.controller.control(obs)
        self.morphology.apply_control(control)

    @classmethod
    def create_agent(cls, body, brain, world):
        morphology = create_soft_body(body, world)
        return Agent(morphology, BaseController.create_controller(morphology.get_input_dim(),
                                                                  morphology.get_output_dim(), brain), world)
