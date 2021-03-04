import time
from copy import deepcopy
import psutil


import ray
from typing import Callable
from train import Agent, StochasticAgent, SacAgent, CnnStochasticActorNetwork, CnnDuelingQNetwork, \
    EpsilonNoiseGenerator, CnnQNetwork
from ExperienceReplay.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from train import Environment
from train import NoiseGenerator, GaussianNoiseGenerator
from train import TD3Agent
from train import DenseContinuousQnetwork, DenseGaussianStochasticActorNetwork, DenseContinuousActorNetwork
from train import CnnContinuousActorNetwork, CnnContinuousQnetwork, CnnGaussianStochasticActorNetwork
from train import Preprocessing
import torch
import numpy as np
from datetime import datetime
import torch.optim as optim
import json



import zmq
import logging
import gym

from visdom_visualizer import Visualizer, VisdomPlot, ImageStream, VideoStream

@ray.remote
class LoggingServer:
    def __init__(self, filename):
        logging.basicConfig(format="%(message)s", level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(filename, 'a+')
        self.logger.addHandler(handler)

    def _time_str(self):
        now = datetime.now()
        time_str = now.strftime("%d/%m/%Y - %H:%M:%S")
        return time_str

    def info(self, process: str, msg: str):
        self.logger.debug("{} -- {} - INFO: {}".format(self._time_str(), process, msg))

    def debug(self, process: str, msg: str):
        self.logger.debug("{} -- {} - DEBUG: {}".format(self._time_str(), process, msg))


""" A dirty merge of all the callbacks required"""
class CallBacks:
    def __init__(self, agent_factory, n_actors, common_config):
        self.viz:Visualizer = Visualizer(context="main")
        # just need the agent factory to get some information about the agent
        agent:Agent = agent_factory()
        self.start_time:datetime = common_config["start_time"]
        loss_names:list = agent.get_loss_values().keys()

        del agent;

        self.loss_lines:dict = dict()
        for loss_name in loss_names:
            self.loss_lines[loss_name] = self.viz.get_line(name="learner_{}".format(loss_name),
                                                      title=loss_name,
                                                      x_label="Batch consumed",
                                                      y_label=loss_name)

        self.learner_speed: VisdomPlot = self.viz.get_line(name="learner_speed",
                                                               title="Learner updates per sec",
                                                               x_label="minutes",
                                                               y_label="Update per sec",
                                                               legend=["Learner 0"])

        self.n_actors = n_actors
        self.actors_speed_line:VisdomPlot = self.viz.get_line(name="actors_speeds",
                                                   title="Actors samples per sec",
                                                   x_label="minutes",
                                                   y_label="Samples per sec",
                                                   legend=["Actor {}".format(i) for i in range(self.n_actors)])

        self.actors_rewards:VisdomPlot = self.viz.get_line(name="actors_rewards",
                                                   title="Actors cumulative rewards",
                                                   x_label="Batch consumed",
                                                   y_label="Cumulative rewards",
                                                   legend=["Actor {}".format(i) for i in range(self.n_actors)])

        self.evaluation_rewards:VisdomPlot = self.viz.get_line(name="evaluation_rewards",
                                                   title="Evaluation cumulative reward",
                                                   x_label="Batch consumed",
                                                   y_label="Cumulative reward",
                                                   legend=["min", "max", "avg"])

        self.evaluation_rewards_time:VisdomPlot = self.viz.get_line(name="evaluation_rewards_time",
                                                   title="Evaluation cumulative reward",
                                                   x_label="minutes",
                                                   y_label="Cumulative reward",
                                                   legend=["min", "max", "avg"])

        self.buffer_usage:VisdomPlot = self.viz.get_line(name="buffer_size",
                                                   title="Buffer usage",
                                                   x_label="minutes",
                                                   y_label="Buffer usage %",
                                                   legend=["usage in %"])

        """self.hardware_usage: VisdomPlot = self.viz.get_line(name="hardware_usage",
                                                            title="Hardware usage",
                                                            x_label="minutes",
                                                            y_label="Percent usage",
                                                            legend=["RAM"] + ["CPU {}".format(i) for i in range(psutil.cpu_count())])"""
        self.hardware_usage: VisdomPlot = self.viz.get_bar(name="hardware_usage",
                                                            title="Hardware usage",
                                                            x_label="Components",
                                                            y_label="Percent usage",
                                                            legend=["RAM"] + ["CPU {}".format(i) for i in
                                                                              range(psutil.cpu_count())])


        self.evaluation_video_stream:ImageStream = self.viz.get_image_stream(name="video_evaluation",
                                                                             width=250,
                                                                             height=250)

    # plot the learner losses
    def plot_losses(self, losses:dict, meta_data:dict):
        x = meta_data["learning_steps"]
        for loss_name, loss_value in losses.items():
            self.loss_lines[loss_name].plot(x=[x], y=[loss_value])

    def plot_learner_speed(self, n_updates_per_sec:float, time:datetime):
        minutes_diff = (time - self.start_time).total_seconds() / 60.0
        x = minutes_diff
        self.learner_speed.plot(x=[x], y=[n_updates_per_sec], var_name="Learner 0")

    def plot_buffer_usage(self, usage:float, time:datetime):
        minutes_diff = (time - self.start_time).total_seconds() / 60.0
        x = minutes_diff
        self.buffer_usage.plot(x=[x], y=[usage])


    def plot_evaluation_rewards(self, rewards:list, meta_data:dict):
        rewards = np.array(rewards)
        x = meta_data["learning_steps"]
        self.evaluation_rewards.plot(x=[x], y=[np.min(rewards)], var_name="min")
        self.evaluation_rewards.plot(x=[x], y=[np.max(rewards)], var_name="max")
        self.evaluation_rewards.plot(x=[x], y=[np.mean(rewards)], var_name="avg")

    def plot_evaluation_rewards_time(self, rewards: list, time:datetime):
        minutes_diff = (time - self.start_time).total_seconds() / 60.0
        x = minutes_diff
        self.evaluation_rewards_time.plot(x=[x], y=[np.min(rewards)], var_name="min")
        self.evaluation_rewards_time.plot(x=[x], y=[np.max(rewards)], var_name="max")
        self.evaluation_rewards_time.plot(x=[x], y=[np.mean(rewards)], var_name="avg")


    def plot_actor_speed(self, actor_id:int, time:datetime, n_samples_per_sec:float):
        minutes_diff = (time - self.start_time).total_seconds() / 60.0
        x = minutes_diff
        self.actors_speed_line.plot(x=[x], y=[n_samples_per_sec], var_name="Actor {}".format(actor_id))

    def plot_hardware_usage(self, time:datetime, mem_usage:float, cpu_usage:list):
        """minutes_diff = (time - self.start_time).total_seconds() / 60.0
        x = minutes_diff
        for id, u in enumerate(cpu_usage):
            self.hardware_usage.plot(x=[x], y=[u], var_name="CPU {}".format(id))
        self.hardware_usage.plot(x=[x], y=[mem_usage], var_name="RAM")"""

        self.hardware_usage.plot(x=[mem_usage] + cpu_usage)


    def plot_actor_rewards(self, actor_id:int, meta_data:dict, rewards:list):
        x = meta_data["learning_steps"]
        self.actors_rewards.plot(x=[x], y=[np.mean(np.array(rewards))], var_name="Actor {}".format(actor_id))

    def evaluation_video(self, frame:np.ndarray):
        self.evaluation_video_stream.push(frame)

@ray.remote
class CallBackServer:
    def __init__(self, callbacks_factory:Callable[[], CallBacks], mappings:dict={}):
        self.mappings = mappings
        self.callbacks = callbacks_factory()

    """def add(self, f_name:str, func:Callable):
        self.mappings[f_name] = func"""

    def callback(self, f_name, **kwargs):
        #self.mappings[f_name](**kwargs)
        self.callbacks.__getattribute__(f_name)(**kwargs)

class Clock:
    def __init__(self, secs:float):
        self.length : float = secs
        self.last_datetime : datetime = datetime.now()

    def count(self) -> bool:
        now: datetime = datetime.now()
        seconds_diff = (now - self.last_datetime).total_seconds()
        if seconds_diff > self.length:
            self.last_datetime = now
            return True
        return False

    def count_time(self)-> tuple :
        now: datetime = datetime.now()
        seconds_diff = (now - self.last_datetime).total_seconds()
        if seconds_diff > self.length:
            self.last_datetime = now
            return (True, seconds_diff)
        return (False, seconds_diff)

@ray.remote
class CentralizedReplayBuffer(object):
    def __init__(self, max_size, alpha, common_config:dict):
        print("test from the centralized replay buffer")

        self.callback_server : CallBackServer = common_config["callback_server"]
        self.buffer = PrioritizedReplayBuffer(max_size, alpha)
        self.logger : LoggingServer = common_config["logger"]
        self.clock:Clock = Clock(secs=5.0)

    def _callbacks(self):
        if self.clock.count():
            self.callback_server.callback.remote(f_name="plot_buffer_usage",
                                                 usage=100.0 * len(self.buffer)/self.buffer.max_size(),
                                                 time=datetime.now())

    def push(self, batch):
        self._callbacks()
        self.buffer.push(*batch)

    def sample(self, batch_size, beta=None, frame_index=None):
        self._callbacks()
        return self.buffer.sample(batch_size, beta, frame_index)

    def update_priorities(self, idxes, priorities):
        self._callbacks()
        self.buffer.update_priorities(idxes=idxes, priorities=priorities)

    def push_priorities(self, batch, priorities):
        self._callbacks()
        for b, priority in zip(batch, priorities):
            state, action, reward, next_state, done = b
            self.buffer.push_priority(state=state,
                                      action=action,
                                      reward=reward,
                                      next_state=next_state,
                                      done=done,
                                      priority=priority)


    def __len__(self):
        return len(self.buffer)

@ray.remote
class Synchronizer(object):
    def __init__(self, n_actors:int):
        self._learner_done:bool = False
        self._n_actors:int = n_actors
        self._actors_done:int = 0
        self._n_pass:int = 0

    def actor_done(self):
        self._actors_done = (self._actors_done % self._n_actors) + 1
        if self._is_learner_done() and self._are_actors_done():
            self._n_pass += 1
            self._actors_done = 0
            self._learner_done = False

    def learner_done(self):
        self._learner_done = True
        if self._is_learner_done() and self._are_actors_done():
            self._n_pass += 1
            self._actors_done = 0
            self._learner_done = False

    def _are_actors_done(self):
        return self._actors_done == self._n_actors

    def _is_learner_done(self):
        return self._learner_done

    def can_pass(self, n_pass):
        return self._n_pass > n_pass, self._n_pass

@ray.remote
class ParametersServer(object):
    def __init__(self):
        self.models:dict = None
        self.meta_data:dict = {}
        self.model_version:int = 0

    def set_models(self, models:dict, meta_data:dict=None):
        self.models = models
        self.meta_data = meta_data
        self.model_version += 1

    def get_models(self, model_version:int):
        if model_version == self.model_version:
            return None, None, self.model_version

        return self.models, self.meta_data, self.model_version


@ray.remote(num_cpus=2, num_gpus=1)
class Learner(object):
    def __init__(self,
                 batch_size: int,
                 n_learning_steps: int,
                 model_update_freq: int,
                 agent_factory: Callable[[], Agent],
                 optimizer_factories: dict,
                 env_factory: Callable[[], Environment],
                 replay_buffer: CentralizedReplayBuffer,
                 param_server: ParametersServer,
                 device: torch.device,
                 common_config: dict,
                 min_replay_size: int = None,
                 gamma: float = 0.99,
                 silent: bool = True):
        self.synchronized : bool = common_config["synchronized"]
        if self.synchronized:
            self.synchronizer : Synchronizer = common_config["synchronizer"]
            self.synchronizer_pass : int = 0
        self.batch_size : int = batch_size
        self.n_learning_steps : int = n_learning_steps
        self.model_update_freq : int = model_update_freq
        self.logger : LoggingServer = common_config["logger"]
        self.callback_server : CallBackServer = common_config["callback_server"]
        self.agent : Agent = agent_factory()
        self.agent.set_optimizers(optimizer_factories=optimizer_factories)
        self.env : Environment = env_factory()
        self.replay_buffer : CentralizedReplayBuffer = replay_buffer
        self.param_server : ParametersServer = param_server
        self.device = device
        self.min_replay_size : int = min_replay_size
        self.silent : bool = silent
        self.gamma : float = gamma
        self.learning_steps : int = 0
        self.update_parameters() # send the parameters a first time

        beta_start : float = 0.4
        beta_frames : int = 10000  # 1000 for classical ; 100000 for atari
        self.beta_by_frame : Callable[[int], float] = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

        self.callback_last_learning_step:int = 0
        self.callback_clock : Clock = Clock(5.0)

        # allows to fetch asynchronously one set of samples in advance
        self.data_fetch_future = None

    def update_criterion(self):
        return self.learning_steps % self.model_update_freq == 0 and self.learning_steps > 0


    def update_parameters(self):
        models : dict = self.agent.get_models()
        models_copy = dict()
        for model_name in models:
            model = models[model_name]["model"]
            models_copy[model_name] = deepcopy(model).to(self.device)

        print("Learner sent parameters.")
        meta_data  : dict = {"learning_steps": self.learning_steps}
        # self.logger.info.remote("Learner", "Parameters sent.")
        self.param_server.set_models.remote(models_copy, meta_data)

    def _fetch_samples(self):
        beta : float = self.beta_by_frame(self.learning_steps)

        if self.data_fetch_future is None:
            self.data_fetch_future = self.replay_buffer.sample.remote(self.batch_size, beta=beta)

        state, action, reward, next_state, \
        done, indices, weights = ray.get(self.data_fetch_future)
        self.data_fetch_future = self.replay_buffer.sample.remote(self.batch_size, beta=beta)

        return state, action, reward, next_state, done, indices, weights

    def learn(self):
        state, action, reward, next_state, done, indices, weights = self._fetch_samples()

        state = torch.from_numpy(state).to(self.device)
        action = torch.from_numpy(action).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        done = torch.from_numpy(done).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)

        # cast to float
        # does so after sending to GPU (possibly)
        if self.env.is_image_based():
            state = state.float()
            next_state = next_state.float()


        batch = {}
        batch["indices"] = indices
        weights = weights
        batch["weights"] = weights
        batch["state"] = state
        batch["action"] = action
        batch["reward"] = reward
        batch["next_state"] = next_state
        batch["done"] = done


        ret = self.agent.learn(sampling="priority", gamma=self.gamma, **batch)
        indices, prios = ret[0], ret[1]

        self.replay_buffer.update_priorities.remote(indices, prios)

        batch.clear()  # safety free

    def stopping_criterion(self):
        return self.learning_steps >= self.n_learning_steps

    def _callbacks(self):
        if self.learning_steps % 25 == 0:
            self.callback_server.callback.remote(f_name="plot_losses",
                                                 losses=self.agent.get_loss_values(),
                                                 meta_data={"learning_steps": self.learning_steps})

        (passed, time_elapsed) = self.callback_clock.count_time()
        if passed:
            n_updates_per_sec = (self.learning_steps - self.callback_last_learning_step)/time_elapsed
            self.callback_server.callback.remote(f_name="plot_learner_speed",
                                                 n_updates_per_sec=n_updates_per_sec,
                                                 time=datetime.now())
            self.callback_last_learning_step = self.learning_steps

    def _synchronize(self):
        if self.learning_steps % self.model_update_freq == 0 and self.learning_steps > 0:
            self.synchronizer.learner_done.remote()
            print("Learner is waiting !")
            t = time.time()
            while True:
                can_pass, n_pass = ray.get(self.synchronizer.can_pass.remote(self.synchronizer_pass))

                if can_pass:
                    self.synchronizer_pass = n_pass
                    break
                else:
                    time.sleep(0.05)
            print("Learner is released after {} sec !".format(time.time() - t))

    def _wait_buffer_min_filled(self):
        while True:
            if ray.get(self.replay_buffer.__len__.remote()) >= self.min_replay_size:
                return

            if self.synchronized:
                self.synchronizer.learner_done.remote()


                can_pass, n_pass = ray.get(self.synchronizer.can_pass.remote(self.synchronizer_pass))

                if can_pass:
                    self.synchronizer_pass = n_pass
                    continue

            time.sleep(0.1)

    def run(self):
        if not self.silent:
            print("Waiting for buffer to be min filled ...")
        self._wait_buffer_min_filled()

        if not self.silent:
            print("Training starts.")
        while True:
            self._callbacks()
            self.learn()
            if self.update_criterion():
                self.update_parameters()

            if self.synchronized:
                self._synchronize()

            if self.silent is False:
                if self.learning_steps % 250 == 0 and self.learning_steps != 0:
                    print("Learning step {}".format(self.learning_steps))
            self.learning_steps += 1


@ray.remote(num_cpus=1)
class Actor(object):
    def __init__(self,
                 actor_id : int,
                 agent_factory: Callable[[], Agent],
                 env_factory: Callable[[], Environment],
                 noise_generator_factory: Callable[[], NoiseGenerator],
                 replay_buffer: CentralizedReplayBuffer,
                 local_buffer_max_size : int,
                 param_server: ParametersServer,
                 device: torch.device,
                 common_config: dict,
                 n_samples: int = None,
                 gamma: float = 0.99,
                 silent: bool = True):
        self.actor_id = actor_id
        self.actor_name = "actor_{}".format(self.actor_id)
        self.synchronized : bool = common_config["synchronized"]
        if self.synchronized:
            self.synchronizer : Synchronizer = common_config["synchronizer"]
            self.synchronizer_pass : int = 0
        self.agent : Agent = agent_factory()
        self.env : Environment = env_factory()
        self.noise_generator : NoiseGenerator = noise_generator_factory()
        self.replay_buffer : CentralizedReplayBuffer = replay_buffer
        self.local_buffer : list = []
        self.local_buffer_max_size : int = local_buffer_max_size
        self.param_server : ParametersServer = param_server
        self.model_version = 0
        self.logger : LoggingServer = common_config["logger"]
        self.callback_server : CallBackServer = common_config["callback_server"]
        self.n_samples : int = n_samples
        self.device : str = device

        if not silent:
            print("Actor {}, parameters loaded.".format(actor_id))

        self.gamma : float = gamma
        self.silent : bool = silent
        self.frame_index : int = 0
        self.param_meta_data : dict = None
        self.callback_last_frame_index : int = self.frame_index
        self.callback_clock = Clock(5.0)
        self.callback_episode_reward : int = 0
        self.callback_end_rewards : list = []
        self.callback_last_model_version : int = -1
        self.callback_last_param_meta_data : dict = None

        while self.model_version == 0:
            self._update_parameters()
            time.sleep(0.1)

    def stopping_criterion(self):
        return self.frame_index >= self.n_samples

    def _update_parameters(self):
        models, param_meta_data, self.model_version = ray.get(self.param_server.get_models.remote(self.model_version))

        if models is not None:
            self.param_meta_data = param_meta_data
            if self.callback_last_param_meta_data is None:
                self.callback_last_param_meta_data = param_meta_data

            models_copy = dict()
            for model_name, model in models.items():
                models_copy[model_name] = deepcopy(model).to(self.device)

            self.agent.update_models(models_copy)
            self.logger.info.remote("Actor {}".format(self.actor_id), "Parameters received.")

            return True

        return False

    def explore(self):
        if isinstance(self.agent, StochasticAgent):
            noisy_policy = lambda state: self.agent.sample_action(state, exploration_noise=self.noise_generator)
        else:
            noisy_policy = lambda state: self.agent.action(state, exploration_noise=self.noise_generator)

        state, next_state, action, reward, done = self.env.experiment(noisy_policy)
        self.callback_episode_reward += reward
        if done:
            self.callback_end_rewards.append(self.callback_episode_reward)
            self.callback_episode_reward = 0

        # float is used once for priorities prediction
        state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        next_state = torch.from_numpy(next_state.astype(np.float32)).unsqueeze(0)

        if self.env.is_discrete():
            action = torch.from_numpy(np.array(action).astype(np.int64))
        else:
            action = torch.from_numpy(np.array(action).astype(np.float32))
        reward = torch.from_numpy(np.array(reward).astype(np.float32))
        done = torch.from_numpy(np.array(done).astype(np.float32))
        weight = torch.from_numpy(np.array(1.0).astype(np.float32))

        priority = self.agent.get_priorities(state,
                                             action.unsqueeze(0),
                                             reward.unsqueeze(0),
                                             next_state,
                                             done.unsqueeze(0),
                                             weight.unsqueeze(0),
                                             self.gamma)
        priority = priority.item()

        # less heavy alternative for buffer
        if self.env.is_image_based():
            state = state.to(dtype=torch.uint8).numpy()
            next_state = next_state.to(dtype=torch.uint8).numpy()
        else: # continuous-vector based
            state = state.to(dtype=torch.float32).numpy()
            next_state = next_state.to(dtype=torch.float32).numpy()

        action = action.numpy()
        reward = reward.numpy()
        done = done.numpy()

        self.local_buffer.append((priority, (state, action, reward, next_state, done)))
        self.frame_index += 1

    def _callback(self):
        (passed, n_sec_elapsed) = self.callback_clock.count_time()
        if passed:
            n_samples_per_sec = (self.frame_index - self.callback_last_frame_index)/n_sec_elapsed
            self.callback_server.callback.remote(f_name="plot_actor_speed",
                                                  actor_id=self.actor_id,
                                                  time=datetime.now(),
                                                  n_samples_per_sec=n_samples_per_sec)
            self.callback_last_frame_index = self.frame_index

            if self.callback_last_model_version < self.model_version:
                if len(self.callback_end_rewards) > 0:
                    self.callback_server.callback.remote(f_name="plot_actor_rewards",
                                                         actor_id=self.actor_id,
                                                         meta_data=self.callback_last_param_meta_data,
                                                         rewards=self.callback_end_rewards)
                    self.callback_end_rewards = []
                self.callback_last_model_version = self.model_version
                self.callback_last_param_meta_data = self.param_meta_data

    def _synchronize(self):
        self.synchronizer.actor_done.remote()
        print("Actor {} is waiting !".format(self.actor_id))
        t = time.time()
        while True:
            can_pass, n_pass = ray.get(self.synchronizer.can_pass.remote(self.synchronizer_pass))

            if can_pass:
                self.synchronizer_pass = n_pass
                break
            else:
                time.sleep(0.05)

        print("Actor {} is released after {} sec !".format(self.actor_id, time.time() - t))

    def run(self):
        with torch.no_grad():
            while not self.stopping_criterion():
                # print("{}: {}".format(self.actor_name, self.frame_index))
                self._callback()
                self._update_parameters()
                self.explore()
                if len(self.local_buffer) >= self.local_buffer_max_size:
                    if not self.silent:
                        print("Actor {} sends local buffer.".format(self.actor_id))
                    (priorities, batch) = zip(*self.local_buffer)
                    self.replay_buffer.push_priorities.remote(batch, priorities)
                    self.local_buffer = list()
                    if self.synchronized:
                        self._synchronize()


            print("{} has generated {} samples.".format(self.actor_name, self.n_samples))

@ray.remote
class Evaluator:
    def __init__(self,
                 agent_factory: Callable[[], Agent],
                 env_factory: Callable[[], Environment],
                 param_server: ParametersServer,
                 device: torch.device,
                 common_config:dict,
                 n_episodes: int = 1,
                 gamma: float = 0.99,
                 silent: bool = True):
        print("EVALUATOR DEVICE {}".format(device))
        self.agent : Agent = agent_factory().to(device)
        self.device : torch.device = device
        self.env : Environment = env_factory()
        self.param_server : ParametersServer = param_server
        self.logger : LoggingServer = common_config["logger"]
        self.callback_server : CallBackServer = common_config["callback_server"]
        self.n_episodes : int = n_episodes # number of episodes per update
        self.gamma : float = gamma
        self.silent : bool = silent
        self.model_version = 0
        self.param_metadata : dict = None

    def _update_parameters(self) -> bool:
        models, self.param_metadata, self.model_version = ray.get(self.param_server.get_models.remote(self.model_version))

        if models is not None:
            models_copy = dict()
            for model_name, model in models.items():
                models_copy[model_name] = deepcopy(model).to(self.device)

            self.agent.update_models(models_copy)

            self.logger.info.remote("Evaluator", "Parameters received.")
            return True # parameters updated

        return False # parameters not updated

    def _get_update_parameters(self):
        while True:
            if self._update_parameters():
                break
            else:
                time.sleep(0.1)

    def _evaluate(self, render=False, render_by_default=False):
        rewards = []
        for i in range(self.n_episodes):
            if self.silent is False:
                print("\rEvaluation episode {}/{}".format(i + 1, self.n_episodes),
                      end="\n" if i == self.n_episodes - 1 else "")
            self.env.reset()
            episode_reward = 0
            done = False
            episode_steps = 0
            while not done:
                if render:
                    self.env.env.render()

                #if render or render_by_default:
                    #time.sleep(1 / 40)

                if isinstance(self.agent, StochasticAgent):
                    policy = lambda state: self.agent.best_action(state)
                else:
                    policy = lambda state: self.agent.action(state)

                #raise Exception()
                """p = self.agent.get_models()["actor-net"]["model"].parameters()
                while True:
                    print(next(p).device)"""

                _, _, _, reward, done = self.env.experiment(policy)
                episode_reward += reward

                episode_steps += 1
                if episode_steps >= self.env.max_steps:
                    break

            if render:
                self.env.env.close()

            if not self.silent:
                print("Episode {} ended after {} steps (max {}) ; reward {}".format(i, episode_steps, self.env.max_steps, episode_reward))

            rewards.append(episode_reward)

        return np.mean(rewards), rewards

    def _callback(self, rewards:list):
        self.callback_server.callback.remote(f_name="plot_evaluation_rewards", rewards=rewards, meta_data=self.param_metadata)
        self.callback_server.callback.remote(f_name="plot_evaluation_rewards_time", rewards=rewards, time=datetime.now())

    def run(self):
        while True:
            self._get_update_parameters()
            _, rewards = self._evaluate()
            self._callback(rewards=rewards)

class Camera:
    def __init__(self, env):
        self.env = env
        pass

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        distance = 10
        yaw = 10
        self.env._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)

@ray.remote
class EvaluatorVideo:
    def __init__(self,
                 agent_factory: Callable[[], Agent],
                 env_factory: Callable[[], Environment],
                 param_server: ParametersServer,
                 device: torch.device,
                 common_config:dict,
                 silent: bool = True):
        self.agent : Agent = agent_factory().to(device)
        self.device : torch.device = device
        self.env : Environment = env_factory()
        if self.env.is_pybullet():
            self.env.env.env._cam_dist = 0.5
        #self.env.env.env._cam_pitch = -90
        #self.env.env._render_width = 500
        self.env.env.reset()
        self.param_server : ParametersServer = param_server
        self.logger : LoggingServer = common_config["logger"]
        self.callback_server : CallBackServer = common_config["callback_server"]
        self.silent : bool = silent
        self.model_version = 0
        self.param_metadata : dict = None
        self._get_update_parameters() # wait until first update

    def _update_parameters(self) -> bool:
        models, self.param_metadata, self.model_version = ray.get(self.param_server.get_models.remote(self.model_version))

        if models is not None:
            models_copy = dict()
            for model_name, model in models.items():
                models_copy[model_name] = deepcopy(model).to(self.device)

            self.agent.update_models(models_copy)

            self.logger.info.remote("Evaluator video", "Parameters received.")
            return True # parameters updated

        return False # parameters not updated

    def _get_update_parameters(self):
        while True:
            if self._update_parameters():
                break
            else:
                time.sleep(0.1)

    def _video_episode(self):
        done = False
        episode_steps = 0
        frames = []
        frame_index = 0
        while not done:
            if isinstance(self.agent, StochasticAgent):
                #policy = lambda state: self.agent.best_action(state)
                def p(state):
                    a = self.agent.best_action(state)
                    #print("Video evaluator action {}".format(a))
                    return a
                policy = p
            else:
                #policy = lambda state: self.agent.action(state)
                def p(state):
                    a = self.agent.action(state)
                    #print("Video evaluator action {}".format(a))
                    return a
                policy = p

            frame, _, _, _, done = self.env.experiment(policy)

            if self.env.is_stacked_states():
                frame = np.expand_dims(frame[:, :, -1], 2)

            if not self.env.is_image_based():
                frame = self.env.env.render(mode="rgb_array")

            frames.append(frame)
            time.sleep(1 / 40)

            episode_steps += 1
            if episode_steps >= self.env.max_steps:
                break
            frame_index += 1

            #print(self.env.env._cam_dist)
            if frame_index % 2 == 0:
                self._callback(np.moveaxis(frame, [0, 1, 2], [1, 2, 0]))


    def _callback(self, frame:np.ndarray):
        self.callback_server.callback.remote(f_name="evaluation_video", frame=frame)

    def run(self):
        while True:
            self._update_parameters()
            self._video_episode()


@ray.remote
class SystemMonitor:
    def __init__(self, callback_server:CallBackServer):
        self.callback_server = callback_server
        self.clock = Clock(5.0)

    def _monitor(self):
        mem = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(percpu=True)
        memory_usage = mem.percent

        return memory_usage, cpu_usage

    def _callback(self):
        if self.clock.count():
            memory_usage, cpu_usage = self._monitor()
            self.callback_server.callback.remote(f_name="plot_hardware_usage",
                                                 time=datetime.now(),
                                                 mem_usage=memory_usage,
                                                 cpu_usage=cpu_usage)

    def run(self):
        while True:
            self._callback()

def apex_parser(filepath):
    with open(filepath) as f:
        data = json.load(f)

    gamma = data["gamma"]

    # env
    env = data["env"]
    env_id = env["name"]
    env_max_steps = env["max_steps"]
    env_pybullet = env["pybullet"]
    env_image_based = env["image_based"]
    env_force_image = env["force_image"]
    env_stacked_states = env["stacked_states"]

    preprocessings = env["preprocessing"]
    preprocessing_funcs = []
    for preprocessing in preprocessings:
        p_name = preprocessing["name"]
        p_parameters = preprocessing["parameters"]
        p_func = lambda state: Preprocessing.__getattribute__(name=p_name)(state, **p_parameters)
        preprocessing_funcs.append(p_func)
    # aggregate
    def preprocessing_func(state):
        func = lambda state: state
        for p in preprocessing_funcs:
            func = lambda state: p(func(state))

        return func(state)

    env_factory: Callable[[], Environment] = lambda: Environment(env=gym.make(env_id),
                                                                 image_state=env_image_based,
                                                                 stack_states=env_stacked_states,
                                                                 max_steps=env_max_steps,
                                                                 state_preprocessing=preprocessing_func,
                                                                 pybullet=env_pybullet,
                                                                 force_image_state=env_force_image,
                                                                 silent=True)
    env_example = env_factory()



    # agent
    agent = data["agent"]
    agent_name = agent["name"]
    agent_config = agent["config"]


class Apex(object):
    def __init(self, ):
        pass


@ray.remote(num_cpus=1, num_gpus=1)
class Sequential:
    def __init__(self,
                 batch_size: int,
                 n_learning_steps: int,
                 agent_factory: Callable[[], Agent],
                 env_factory: Callable[[], Environment],
                 optimizer_factories: dict,
                 noise_generator_factory: Callable[[], NoiseGenerator],
                 replay_buffer_factory: Callable[[], PrioritizedReplayBuffer],
                 param_server: ParametersServer,
                 device: torch.device,
                 common_config: dict,
                 min_replay_size: int = None,
                 gamma: float = 0.99,
                 silent: bool = True):
        self.batch_size : int = batch_size
        self.n_learning_steps : int = n_learning_steps
        self.logger : LoggingServer = common_config["logger"]
        self.callback_server : CallBackServer = common_config["callback_server"]
        self.agent : Agent = agent_factory()
        self.env = env_factory()
        self.noise_generator : NoiseGenerator = noise_generator_factory()
        print(next(self.agent.get_models()["q-net-1"]["model"].parameters()).device)
        self.agent.set_optimizers(optimizer_factories=optimizer_factories)
        self.replay_buffer : PrioritizedReplayBuffer = replay_buffer_factory()
        self.param_server : ParametersServer = param_server
        self.device : torch.device  = device
        self.min_replay_size : int = min_replay_size
        self.silent : bool = silent
        self.gamma : float = gamma
        self.learning_steps : int = 0
        self.frame_index : int = 0

        beta_start : float = 0.4
        beta_frames : int = 10000  # 1000 for classical ; 100000 for atari
        self.beta_by_frame : Callable[[int], float] = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

        self.callback_episode_reward:int = 0
        self.callback_end_rewards:list = []
        self.callback_last_learning_step:int = 0
        self.callback_clock = Clock(5.0)

    def learn(self):
        beta : float = self.beta_by_frame(self.learning_steps)
        state, action, reward, next_state,\
        done, indices, weights = self.replay_buffer.sample(self.batch_size, beta=beta)
        state = torch.from_numpy(state).to(self.device)
        action = torch.from_numpy(action).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        done = torch.from_numpy(done).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)

        # cast to float
        # does so after sending to GPU (possibly)
        if self.env.is_image_based():
            state = state.float()
            next_state = next_state.float()

        batch = {}
        batch["indices"] = indices
        weights = weights
        batch["weights"] = weights
        batch["state"] = state
        batch["action"] = action
        batch["reward"] = reward
        batch["next_state"] = next_state
        batch["done"] = done

        ret = self.agent.learn(sampling="priority", gamma=self.gamma, **batch)
        indices, prios = ret[0], ret[1]

        self.replay_buffer.update_priorities(indices, prios)

        batch.clear()  # safety free

    def explore(self):
        if isinstance(self.agent, StochasticAgent):
            noisy_policy = lambda state: self.agent.sample_action(torch.from_numpy(state).to(self.device), exploration_noise=self.noise_generator)
        else:
            noisy_policy = lambda state: self.agent.action(state, exploration_noise=self.noise_generator)

        state, next_state, action, reward, done = self.env.experiment(noisy_policy)
        self.callback_episode_reward += reward
        if done:
            self.callback_end_rewards.append(self.callback_episode_reward)
            self.callback_episode_reward = 0

        if self.env.is_image_based():
            state = np.stack([state]).astype(np.uint8)
            next_state = np.stack([next_state]).astype(np.uint8)
        else: # continous-vector based
            state = np.stack([state]).astype(np.float32)
            next_state = np.stack([next_state]).astype(np.float32)

        if self.env.is_discrete():
            action = np.array(action).astype(np.int64)
        else:
            action = np.array(action).astype(np.float32)
        reward = np.array(reward).astype(np.float32)
        done = np.array(done).astype(np.float32)

        self.replay_buffer.push(*(state, action, reward, next_state, done))
        self.frame_index += 1

    def update_criterion(self):
        return self.learning_steps % 500 == 0 and self.learning_steps > 0

    def update_parameters(self):
        models : dict = self.agent.get_models()
        models_copy = dict()
        for model_name in models:
            model = models[model_name]["model"]
            models_copy[model_name] = deepcopy(model).to(self.device)

        print("Learner sent parameters.")
        meta_data  : dict = {"learning_steps": self.learning_steps}
        # self.logger.info.remote("Learner", "Parameters sent.")
        self.param_server.set_models.remote(models_copy, meta_data)


    def stopping_criterion(self):
        return self.learning_steps >= self.n_learning_steps


    """
        - every n steps: list of rewards
        
    """
    def _callbacks(self):
        if self.learning_steps % 25 == 0:
            self.callback_server.callback.remote(f_name="plot_losses",
                                                 losses=self.agent.get_loss_values(),
                                                 meta_data={"learning_steps": self.learning_steps})

            if len(self.callback_end_rewards) > 0:
                self.callback_server.callback.remote(f_name="plot_actor_rewards",
                                                     actor_id=0,
                                                     meta_data={"learning_steps": self.learning_steps},
                                                     rewards=self.callback_end_rewards)
                self.callback_end_rewards = list()

        (passed, time_elapsed) = self.callback_clock.count_time()
        if passed:
            n_updates_per_sec = (self.learning_steps - self.callback_last_learning_step)/time_elapsed
            self.callback_server.callback.remote(f_name="plot_buffer_usage",
                                                 usage=100.0 * len(self.replay_buffer)/self.replay_buffer.max_size(),
                                                 time=datetime.now())

            self.callback_server.callback.remote(f_name="plot_learner_speed",
                                                 n_updates_per_sec=n_updates_per_sec,
                                                 time=datetime.now())
            self.callback_last_learning_step = self.learning_steps


    def _evaluate_criterion(self):
        return self.learning_steps % self.evaluate_every_n_steps == 0

    def run(self):
        if not self.silent:
            print("Waiting for buffer to be min filled ...")


        while len(self.replay_buffer) < self.min_replay_size:
            print("\rSamples {}/{} ".format(len(self.replay_buffer), self.min_replay_size), \
                  end="\n" if len(self.replay_buffer) == self.min_replay_size else "")

            self.explore()

        if not self.silent:
            print("Training starts.")

        while not self.stopping_criterion():
            if self.update_criterion():
                self.update_parameters()

            self._callbacks()
            self.learn()
            self.explore()

            if self.silent is False:
                if self.learning_steps % 250 == 0 and self.learning_steps != 0:
                    print("Learning step {}".format(self.learning_steps))

            self.learning_steps += 1

def apex():
    batch_size = 512
    max_replay_size = 12500#250000
    gamma = 0.99
    min_replay_size = 1000  # min number of samples before starting training
    max_steps = 10000  # max steps in the env
    alpha_replay = 0.6
    silent = False
    learner_device = torch.device("cuda")  # if torch.cuda.is_available() else "cpu")
    actor_device = torch.device("cpu")
    n_samples_per_actor = 10000000  # number of tasked samples per actor
    update_freq_local_buffer_size_ratio = 5 # for synchronized mode only
    model_update_freq = 150 # learning steps between each learner update (sending to ParamServer)
    #local_buffer_max_size = 500  # actor buffer size (before flush)
    local_buffer_max_size = model_update_freq * update_freq_local_buffer_size_ratio  # actor buffer size (before flush)
    n_learning_steps = 100000  # learner steps
    n_episodes_eval = 1
    n_actors = 3
    logging_file = "ray_log_test2.log"
    start_time = datetime.now()
    synchronized = True

    common_config = {"start_time": start_time}

    logging = LoggingServer.remote(filename=logging_file)

    common_config["logger"] = logging

    synchronizer = Synchronizer.remote(n_actors=n_actors)
    common_config["synchronizer"] = synchronizer
    common_config["synchronized"] = synchronized

    env_id = "Pong-v0"#"ReacherPyBulletEnv-v0"  # "Pendulum-v0" #"Pong-v0"####"CartPole-v0"#"Assault-v0"#"SpaceInvadersNoFrameskip-v4" #"CartPole-v0"###"SpaceInvadersNoFrameskip-v4"#"Acrobot-v1"#"MountainCar-v0"#"CartPole-v0"#"CartPole-v0"#"CartPole-v0"# "PongNoFrameskip-v4"#"Pong-v0" ##
    pybullet = False
    preprocessing = lambda image: Preprocessing.color_to_grayscale(Preprocessing.resize(image, 0.5))
    env_factory: Callable[[], Environment] = lambda: Environment(env=gym.make(env_id),
                                                                 image_state=True,
                                                                 stack_states=3,
                                                                 max_steps=max_steps,
                                                                 state_preprocessing=preprocessing,
                                                                 pybullet=pybullet,
                                                                 force_image_state=False,
                                                                 silent=True)
    env = env_factory()
    print(env.get_state_space())
    """critic_model_factory = lambda: DenseContinuousQnetwork(state_size=env.get_state_space()[0],
                                                           action_dim=env.get_action_space()["shape"],
                                                           hidden_size=256)

    actor_model_factory = lambda: DenseContinuousActorNetwork(state_size=env.get_state_space()[0],
                                                              action_dim=env.get_action_space()["shape"],
                                                              hidden_size=256)"""

    """critic_model_factory = lambda: CnnContinuousQnetwork(state_size=env.get_state_space(),
                                                        action_dim=env.get_action_space()["shape"],
                                                       hidden_size=256)"""

    """actor_model_factory = lambda: CnnContinuousActorNetwork(state_size=env.get_state_space(),
                                                          action_dim=env.get_action_space()["shape"],
                                                          hidden_size=256)"""

    """actor_model_factory = lambda: CnnGaussianStochasticActorNetwork(state_size=env.get_state_space(),
                                                                    action_dim=env.get_action_space()["shape"],
                                                                    hidden_size=256)"""

    actor_model_factory = lambda: CnnStochasticActorNetwork(state_size=env.get_state_space(),
                                                            action_dim=env.get_action_space()["n_actions"],
                                                            hidden_size=256)
    critic_model_factory = lambda: CnnQNetwork(state_size=env.get_state_space(),
                                                            action_dim=env.get_action_space()["n_actions"],
                                                            hidden_size=256)
    """agent_factory: Callable[[torch.device], Agent] = lambda device: TD3Agent(noise_clip=0.5,
                                                                             policy_noise=0.2,
                                                                             actor_freq=2,
                                                                             update_polyak=0.99,
                                                                             actor_model_factory=actor_model_factory,
                                                                             critic_model_factory=critic_model_factory).to(device)"""

    agent_factory : Callable[[torch.device], Agent] = lambda device: SacAgent(action_dim=env.get_action_space()["n_actions"],
                                                                         alpha="auto",
                                                                         tau=0.99,  # 100 for classical ; 1000 for atari
                                                                         discrete_actions=True,
                                                                         actor_model_factory=actor_model_factory,
                                                                         critic_model_factory=critic_model_factory).to(device)

    agent_factory_actor: Callable[[], Agent] = lambda: agent_factory(actor_device)
    agent_factory_learner: Callable[[], Agent] = lambda: agent_factory(learner_device)

    callbacks: CallBacks = CallBacks(agent_factory=agent_factory_actor, n_actors=n_actors, common_config=common_config)
    callback_server: CallBackServer = CallBackServer.remote(lambda: CallBacks(agent_factory=agent_factory_actor,
                                                                              n_actors=n_actors,
                                                                              common_config=common_config))
    # callback_server.add("line_losses", callbacks.plot_losses)

    hw_monitor = SystemMonitor.remote(callback_server=callback_server)

    common_config["callback_server"] = callback_server

    replay_buffer: CentralizedReplayBuffer = CentralizedReplayBuffer. \
        options(name="ReplayBuffer", lifetime="detached").remote(max_size=max_replay_size, alpha=alpha_replay,
                                                                 common_config=common_config)

    param_server: ParametersServer = ParametersServer.options(name="ParamServer", lifetime="detached").remote()

    optimizer_factory_critic = lambda model: optim.Adam(model.parameters(), lr=5e-5)
    optimizer_factory_actor = lambda model: optim.Adam(model.parameters(), lr=5e-5)
    optimizer_factory_alpha = lambda log_alpha: optim.Adam([log_alpha], lr=1e-4)

    """optimizer_factories = {"q-net-1": optimizer_factory_critic,
                           "q-net-2": optimizer_factory_critic,
                           "actor-net": optimizer_factory_actor}"""
    optimizer_factories = {"soft-q-net-1": optimizer_factory_critic,
                           "soft-q-net-2": optimizer_factory_critic,
                           "actor-net": optimizer_factory_actor}
    optimizer_factories["log_alpha"] = optimizer_factory_alpha

    """noise_generator_factory: Callable[[], NoiseGenerator] = lambda: GaussianNoiseGenerator(seed=0, low=-1, high=1,
                                                                                           max_sigma=0.5, min_sigma=0.5)
    """
    noise_generator_factory: Callable[[], NoiseGenerator] = lambda: EpsilonNoiseGenerator(action_dim=env.get_action_space()["n_actions"],
                                                                                          epsilon_start=1.0,
                                                                                          epsilon_final=0.01,
                                                                                          epsilon_decay=100000)

    actor_factory = lambda id: Actor.options(name="Actor{}".format(id), lifetime="detached"). \
        remote(common_config=common_config,
               actor_id=id,
               agent_factory=agent_factory_actor,
               env_factory=env_factory,
               noise_generator_factory=noise_generator_factory,
               replay_buffer=replay_buffer,
               local_buffer_max_size=local_buffer_max_size,
               param_server=param_server,
               device=actor_device,
               n_samples=n_samples_per_actor,
               gamma=gamma,
               silent=True)

    actors = [actor_factory(i) for i in range(n_actors)]

    learner = Learner.options(name="Learner", lifetime="detached").remote(common_config=common_config,
                                                                          batch_size=batch_size,
                                                                          n_learning_steps=n_learning_steps,
                                                                          model_update_freq=model_update_freq,
                                                                          agent_factory=agent_factory_learner,
                                                                          optimizer_factories=optimizer_factories,
                                                                          env_factory=env_factory,
                                                                          replay_buffer=replay_buffer,
                                                                          param_server=param_server,
                                                                          device=learner_device,
                                                                          min_replay_size=min_replay_size,
                                                                          gamma=gamma,
                                                                          silent=False)

    evaluator = Evaluator.options(name="Evaluator", lifetime="detached").remote(common_config=common_config,
                                                                                agent_factory=agent_factory_actor,
                                                                                env_factory=env_factory,
                                                                                param_server=param_server,
                                                                                device=actor_device,
                                                                                n_episodes=n_episodes_eval,
                                                                                gamma=gamma,
                                                                                silent=False)
    video_evaluator = EvaluatorVideo.options(name="EvaluatorVideo", lifetime="detached").remote(
        common_config=common_config,
        agent_factory=agent_factory_actor,
        env_factory=env_factory,
        param_server=param_server,
        device=actor_device,
        silent=False)

    ray.wait([actor.run.remote() for actor in actors] + \
             [learner.run.remote()] + \
             [evaluator.run.remote()] + \
             [hw_monitor.run.remote()]+ \
             [video_evaluator.run.remote()])




def sequential():
    batch_size = 512
    max_replay_size = 250000
    gamma = 0.99
    min_replay_size = 1000  # min number of samples before starting training
    max_steps = 10000  # max steps in the env
    alpha_replay = 0.6
    silent = False
    learner_device = torch.device("cpu")  # if torch.cuda.is_available() else "cpu")
    actor_device = torch.device("cpu")
    n_samples_per_actor = 10000000  # number of tasked samples per actor
    local_buffer_max_size = 500  # actor buffer size (before flush)
    n_learning_steps = 100000  # learner steps
    n_episodes_eval = 50
    n_actors = 3
    logging_file = "ray_log_test2.log"
    start_time = datetime.now()
    common_config = {"start_time": start_time}

    logging = LoggingServer.remote(filename=logging_file)

    common_config["logger"] = logging

    env_id = "CarRacing-v0"  # "ReacherPyBulletEnv-v0"  # "Pendulum-v0" #"Pong-v0"####"CartPole-v0"#"Assault-v0"#"SpaceInvadersNoFrameskip-v4" #"CartPole-v0"###"SpaceInvadersNoFrameskip-v4"#"Acrobot-v1"#"MountainCar-v0"#"CartPole-v0"#"CartPole-v0"#"CartPole-v0"# "PongNoFrameskip-v4"#"Pong-v0" ##
    pybullet = False
    preprocessing = lambda image: Preprocessing.color_to_grayscale(Preprocessing.resize(image, 1.0))
    env_factory: Callable[[], Environment] = lambda: Environment(env=gym.make(env_id),
                                                                 image_state=True,
                                                                 stack_states=1,
                                                                 max_steps=max_steps,
                                                                 state_preprocessing=preprocessing,
                                                                 pybullet=pybullet,
                                                                 force_image_state=False,
                                                                 silent=True)
    env = env_factory()
    print(env.get_state_space())
    """critic_model_factory = lambda: DenseContinuousQnetwork(state_size=env.get_state_space()[0],
                                                           action_dim=env.get_action_space()["shape"],
                                                           hidden_size=256)

    actor_model_factory = lambda: DenseContinuousActorNetwork(state_size=env.get_state_space()[0],
                                                              action_dim=env.get_action_space()["shape"],
                                                              hidden_size=256)"""

    critic_model_factory = lambda: CnnContinuousQnetwork(state_size=env.get_state_space(),
                                                         action_dim=env.get_action_space()["shape"],
                                                         hidden_size=256)

    actor_model_factory = lambda: CnnContinuousActorNetwork(state_size=env.get_state_space(),
                                                            action_dim=env.get_action_space()["shape"],
                                                            hidden_size=256)

    agent_factory: Callable[[torch.device], Agent] = lambda device: TD3Agent(noise_clip=0.5,
                                                                             policy_noise=0.2,
                                                                             actor_freq=2,
                                                                             update_polyak=0.99,
                                                                             actor_model_factory=actor_model_factory,
                                                                             critic_model_factory=critic_model_factory).to(device)

    agent_factory_actor: Callable[[], Agent] = lambda: agent_factory(actor_device)
    agent_factory_learner: Callable[[], Agent] = lambda: agent_factory(learner_device)

    callback_server: CallBackServer = CallBackServer.remote(lambda: CallBacks(agent_factory=agent_factory_actor,
                                                                              n_actors=n_actors,
                                                                              common_config=common_config))
    # callback_server.add("line_losses", callbacks.plot_losses)

    hw_monitor = SystemMonitor.remote(callback_server=callback_server)

    common_config["callback_server"] = callback_server

    replay_buffer_factory = lambda: PrioritizedReplayBuffer(size=max_replay_size, alpha=alpha_replay)

    param_server: ParametersServer = ParametersServer.options(name="ParamServer", lifetime="detached").remote()

    optimizer_factory_critic = lambda model: optim.Adam(model.parameters(), lr=1e-3)
    optimizer_factory_actor = lambda model: optim.Adam(model.parameters(), lr=1e-3)
    optimizer_factory_alpha = lambda log_alpha: optim.Adam([log_alpha], lr=1e-3)

    optimizer_factories = {"q-net-1": optimizer_factory_critic,
                           "q-net-2": optimizer_factory_critic,
                           "actor-net": optimizer_factory_actor}

    noise_generator_factory: Callable[[], NoiseGenerator] = lambda: GaussianNoiseGenerator(seed=0, low=-1, high=1,
                                                                                           max_sigma=0.1, min_sigma=0.1)


    sequential_learner = Sequential.options(name="learner", lifetime="detached").\
                             remote(common_config=common_config,
                                    batch_size=batch_size,
                                    n_learning_steps=n_learning_steps,
                                    replay_buffer_factory=replay_buffer_factory,
                                    env_factory=env_factory,
                                    noise_generator_factory=noise_generator_factory,
                                    agent_factory=agent_factory_learner,
                                    optimizer_factories=optimizer_factories,
                                    param_server=param_server,
                                    device=learner_device,
                                    min_replay_size=min_replay_size,
                                    gamma=gamma,
                                    silent=False)

    evaluator = Evaluator.options(name="Evaluator", lifetime="detached").remote(common_config=common_config,
                                                                                agent_factory=agent_factory_actor,
                                                                                env_factory=env_factory,
                                                                                param_server=param_server,
                                                                                device=actor_device,
                                                                                n_episodes=n_episodes_eval,
                                                                                gamma=gamma,
                                                                                silent=False)
    video_evaluator = EvaluatorVideo.options(name="EvaluatorVideo", lifetime="detached").remote(
        common_config=common_config,
        agent_factory=agent_factory_actor,
        env_factory=env_factory,
        param_server=param_server,
        device=actor_device,
        silent=False)

    ray.wait([sequential_learner.run.remote()] +\
             [evaluator.run.remote()] + \
             [hw_monitor.run.remote()] + \
             [video_evaluator.run.remote()])

ray.init()
if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.is_available())
    print(torch.cuda.is_available())
    print(torch.cuda.is_available())
    print(torch.cuda.is_available())
    apex()



    # synchronize one actor and learner
