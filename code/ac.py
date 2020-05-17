import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
from collections import deque
from dialogue_config import rule_requests, agent_actions
import re

class ActorCritic:
	def __init__(self,state_size, constants, sess):
		self.sess = sess
		self.hyper_parameters = constants['agent']
		self.memory = []
		self.index_memory = 0
		self.max_memory_size = self.hyper_parameters['max_memory_size']
		self.init_epsilon = self.hyper_parameters['init_epsilon']
		self.vanilla = self.hyper_parameters['vanilla']
		self.learning_rate = self.hyper_parameters['learning_rate']
		self.gamma = self.hyper_parameters['gamma']
		self.batch_size = self.hyper_parameters['batch_size']
		self.hidden_units = self.hyper_parameters['dqn_hidden_units']
		self.epsilon_decay=self.hyper_parameters['epsilon_decay']
		self.tau=self.hyper_parameters['tau']
		self.load_weights_file_path = self.hyper_parameters['load_weights_file_path']
		self.save_weights_file_path = self.hyper_parameters['save_weights_file_path']
		
		if self.max_memory_size < self.batch_size:
			raise ValueError('Batch size should be less than memory size')
		self.state_size = state_size
		self.possible_actions = agent_actions
		self.num_actions = len(self.possible_actions)
		self.rule_request_set = rule_requests
		self.reset()

		self.memory = deque(maxlen=2000)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32, 
			[None, self.num_actions]) 

		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output, 
			actor_model_weights, -self.actor_critic_grad) 
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output, 
			self.critic_action_input) 
		self.sess.run(tf.initialize_all_variables())

	def map_index_to_action(self, index):
		for (i, action) in enumerate(self.possible_actions):
			if index == i:
				return copy.deepcopy(action)
			raise ValueError('Index: {} not in range of possible actions'.format(index))

	def empty_memory(self):
		self.memory = []
		self.index_memory = 0

	def is_memory_full(self):
		return len(self.memory) == self.max_memory_size

	def create_actor_model(self):
		state_input = Input(shape=(self.batch_size, self.state_size))
		h1 = Dense(self.hidden_units, activation='relu')(state_input)
		h2 = Dense(48, activation='relu')(h1)
		h3 = Dense(24, activation='relu')(h2)
		output = Dense(self.num_actions, activation='relu')(h1)
		
		model = Model(input=state_input, output=output)
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model

	def create_critic_model(self):
		state_input = Input(shape=state_size)
		state_h1 = Dense(24, activation='relu')(state_input)
		state_h2 = Dense(48)(state_h1)
		
		action_input = Input(shape=num_actions)
		action_h1    = Dense(48)(action_input)
		
		merged    = Add()([state_h2, action_h1])
		merged_h1 = Dense(24, activation='relu')(merged)
		output = Dense(1, activation='relu')(merged_h1)
		model  = Model(input=[state_input,action_input], output=output)
		
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, _ = sample
			predicted_action = self.actor_model.predict(cur_state)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_state,
				self.critic_action_input: predicted_action
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state,
				self.actor_critic_grad: grads
			})
            
	def _train_critic(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
			if not done:
				target_action = self.target_actor_model.predict(new_state)
				future_reward = self.target_critic_model.predict(
					[new_state, target_action])[0][0]
				reward += self.gamma * future_reward
			self.critic_model.fit([cur_state, action], reward, verbose=0)
		
	def train(self):
		batch_size = 32
		if len(self.memory) < batch_size:
			return

		rewards = []
		samples = random.sample(self.memory, batch_size)
		self._train_critic(samples)
		self._train_actor(samples)


	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]
		self.target_critic_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.critic_target_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]
		self.critic_target_model.set_weights(critic_target_weights)		

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()


	def act(self, cur_state):
		self.init_epsilon *= self.epsilon_decay
		if np.random.random() < self.init_epsilon:
			index = random.randint(0, self.num_actions - 1)
			action = self.map_index_to_action()

			return index
		return self.actor_model.predict(cur_state)

	def reset(self):
		self.rule_current_slot_index = 0
		self.rule_phase = 'not done'




