#!/usr/bin/env python
# coding: utf-8

import gym
import torch
import time
import os
import ray
import numpy as np
import pandas as pd

from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt

from memory_remote import ReplayBuffer_remote
from custom_cartpole import CartPoleEnv

FloatTensor = torch.FloatTensor


ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}


class DQN_agent(object):
    def __init__(self, env, hyper_params, action_space=len(ACTION_DICT)):

        self.env = env
        self.max_episode_steps = env._max_episode_steps

        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space

        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate=hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)

        self.memory = ReplayBuffer(hyper_params['memory_size'])

        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']

    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(self, state):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon,
                                       self.final_epsilon,
                                       self.steps,
                                       self.epsilon_decay_steps)

        if p < epsilon:
            # return action
            return randint(0, self.action_space - 1)
        else:
            # return action
            return self.greedy_policy(state)

    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    def update_batch(self):
        pass

    def learn(self):
        pass


@ray.remote
class DQN_model_server(DQN_agent):
    def __init__(self, env, hyper_params, memory_server, action_space=len(ACTION_DICT)):
        super().__init__(env, hyper_params, action_space)
        self.memory_server = memory_server

    def update_batch(self):
        batch = self.memory_server.sample(self.batch_size)
        if not batch:
            return

        (states, actions, reward, next_states, is_terminal) = batch

        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size, dtype=torch.long)

        # Current Q Values --- Predict the Q-value from the 'eval_model' based on (states, actions)
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]

        # Calculate target --- Predict the Q-value from the 'target model' based on (next_states), and take max of each Q-value vector, Q_max
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)

        q_next = q_next[batch_index, actions]
        q_target = FloatTensor([reward[index] if is_terminal[index] else reward[index] + self.beta * q_next[index] for index in range(self.batch_size)])

        # update model
        self.eval_model.fit(q_values, q_target)

    def learn(self):
        if self.steps % self.update_steps == 0:
            self.update_batch()
        if self.use_target_model:
            if self.steps % self.model_replace_freq == 0:
                self.target_model.replace(self.eval_model)

        self.steps += 1

    def get_eval_model(self):
        return self.eval_model


class DQN_memory_server(object):
    def __init__(self, memory_size):
        self.memory_server = ReplayBuffer_remote.remote(memory_size)

    def add(self, obs_t, action, reward, obs_tp1, done):
        self.memory_server.add.remote(obs_t, action, reward, obs_tp1, done)

    def sample(self, batch_size):
        return ray.get(self.memory_server.sample.remote(batch_size))


@ray.remote
def collecting_worker(model_server, memory_server, env, test_interval_per_cw):
    for _ in tqdm(range(test_interval_per_cw), desc="Training"):
        state = env.reset()
        done = False
        steps = 0

        eval_model = ray.get(model_server.get_eval_model.remote())
        while steps < env._max_episode_steps and not done:
            action = ray.get(model_server.explore_or_exploit_policy.remote(state))
            next_state, reward, done, _ = env.step(action)
            memory_server.add(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            model_server.learn.remote()


@ray.remote
def evaluation_worker(model_server, env, trials_per_ew):
    total_reward = 0
    for _ in tqdm(range(trials_per_ew), desc="Evaluating"):
        state = env.reset()
        done = False
        steps = 0

        while steps < env._max_episode_steps and not done:
            steps += 1
            action = ray.get(model_server.greedy_policy.remote(state))
            state, reward, done, _ = env.step(action)
            total_reward += reward

    avg_reward = total_reward / trials_per_ew

    return avg_reward


class distributed_DQN_agent(object):
    def __init__(self, env, hyper_params, cw_num, ew_num, training_episodes, test_interval, trials):
        self.env = env
        self.memory_server = DQN_memory_server(hyper_params['memory_size'])
        self.model_server = DQN_model_server.remote(env, hyper_params, self.memory_server)
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.training_episodes = training_episodes
        self.test_interval = test_interval
        self.trials = trials

    def learn_and_evaluate(self):
        test_number = self.training_episodes // self.test_interval
        all_results = []

        for index in range(test_number):
            print('index: {0} of {1}'.format(index, test_number))
            # Initialization
            cw_indices = []
            ew_indices = []

            # Learn
            for _ in range(self.cw_num):
                cw_index = collecting_worker.remote(self.model_server, self.memory_server, self.env, self.test_interval // self.cw_num)
                cw_indices.append(cw_index)

            # Evaluate
            for _ in range(self.ew_num):
                ew_index = evaluation_worker.remote(self.model_server, self.env, self.trials // self.ew_num)
                ew_indices.append(ew_index)

            total_reward = sum(ray.get(ew_indices))
            avg_reward = total_reward / self.ew_num
            all_results.append(avg_reward)

        return all_results
