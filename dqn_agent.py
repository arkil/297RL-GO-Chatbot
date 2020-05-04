from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random, copy
import numpy as np
from dialogue_config import rule_requests, agent_actions
import re


class DQNAgent:
    """The DQN agent that interacts with the user."""

    def __init__(self, state_size, constants):
        """
        Parameters:
            state_size (int): The state representation size or length of numpy array
            constants (dict): Dictionary of constants

        """
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

        self.load_weights_file_path = self.hyper_parameters['load_weights_file_path']
        self.save_weights_file_path = self.hyper_parameters['save_weights_file_path']

        if self.max_memory_size < self.batch_size:
            raise ValueError('Batch size should be less than memory size')

        self.state_size = state_size
        self.possible_actions = agent_actions
        self.num_actions = len(self.possible_actions)
        self.rule_request_set = rule_requests
        self.behaviour_model = self.build_model()
        self.target_model = self.build_model()
        self.load_weights()

        self.reset()

    def build_model(self):
        """Builds and returns model/graph of neural network."""

        model = Sequential()
        model.add(Dense(self.hidden_units, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def reset(self):
        """Resets the rule-based variables."""
        self.rule_current_slot_index = 0
        self.rule_phase = 'not done'

    def get_action(self, state, use_rule=False):
        """
        Returns action of agent based on rule-based or neural network 
        Parameters:
            state (numpy.array): The database with format dict(long: dict)
            use_rule (bool): Depends on which phase which action to chose(default :false)

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself
        """
        if self.init_epsilon > random.random():
            index = random.randint(0, self.num_actions - 1)
            action = self.map_index_to_action(index)
            return index, action
        else:
            if use_rule:
                return self.rule_action()
            else:
                return self.dqn_action(state)

    def rule_action(self):
        """
        Returns a rule-based policy action.
        Selects the next action of a simple rule-based policy.
        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself
        """

        if self.rule_current_slot_index < len(self.rule_request_set):
            slot = self.rule_request_set[self.rule_current_slot_index]
            self.rule_current_slot_index += 1
            rule_response = {'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}}
        elif self.rule_phase == 'not done':
            rule_response = {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
            self.rule_phase = 'done'
        elif self.rule_phase == 'done':
            rule_response = {'intent': 'done', 'inform_slots': {}, 'request_slots': {}}
        else:
            raise Exception('Should not have reached this clause')

        index = self.map_action_to_index(rule_response)
        return index, rule_response

    def map_action_to_index(self, response):
        """
        Maps an action to an index from possible actions.

        Parameters:
            response (dict)

        Returns:
            int
        """

        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i
        raise ValueError('Response: {} not found in possible actions'.format(response))

    def dqn_action(self, state):
        """
        Returns a behavior model output given a state.

        Parameters:
            state (numpy.array)

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself
        """

        index = np.argmax(self.dqn_predict_one(state))
        action = self.map_index_to_action(index)
        return index, action

    def map_index_to_action(self, index):
        """
        Maps an index to an action in possible actions.

        Parameters:
            index (int)

        Returns:
            dict
        """

        for (i, action) in enumerate(self.possible_actions):
            if index == i:
                return copy.deepcopy(action)
        raise ValueError('Index: {} not in range of possible actions'.format(index))

    def dqn_predict_one(self, state, target=False):
        """
        Returns a model prediction given a state.

        Parameters:
            state (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        return self.dqn_predict(state.reshape(1, self.state_size), target=target).flatten()

    def dqn_predict(self, states, target=False):
        """
        Returns a model prediction given an array of states.

        Parameters:
            states (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        if target:
            return self.target_model.predict(states)
        else:
            return self.behaviour_model.predict(states)

    def add_experience(self, state, action, reward, next_state, done):
        """
        Adds an experience tuple made of the parameters to the memory.

        Parameters:
            state (numpy.array)
            action (int)
            reward (int)
            next_state (numpy.array)
            done (bool)

        """

        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.index_memory] = (state, action, reward, next_state, done)
        self.index_memory = (self.index_memory + 1) % self.max_memory_size

    def empty_memory(self):
        """Empties the memory and resets the memory index."""

        self.memory = []
        self.index_memory = 0

    def is_memory_full(self):
        """Returns true if the memory is full."""

        return len(self.memory) == self.max_memory_size

    def train(self):
        """
        Trains the agent by improving the behavior model given the memory tuples.

        Takes batches of memories from the memory pool and processing them. The processing takes the tuples and stacks
        them in the correct format for the neural network and calculates the Bellman equation for Q-Learning.

        """

        # Calc. num of batches to run
        num_batches = len(self.memory) // self.batch_size
        for b in range(num_batches):
            batch = random.sample(self.memory, self.batch_size)

            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])

            assert states.shape == (self.batch_size, self.state_size), 'States Shape: {}'.format(states.shape)
            assert next_states.shape == states.shape

            behavior_state_preds = self.dqn_predict(states)  # For leveling error
            if not self.vanilla:
                behavior_next_states_preds = self.dqn_predict(next_states)  # For indexing for DDQN
            target_next_state_preds = self.dqn_predict(next_states, target=True)  # For target value for DQN (& DDQN)

            inputs = np.zeros((self.batch_size, self.state_size))
            targets = np.zeros((self.batch_size, self.num_actions))

            for i, (s, a, r, s_, d) in enumerate(batch):
                t = behavior_state_preds[i]
                if not self.vanilla:
                    t[a] = r + self.gamma * target_next_state_preds[i][np.argmax(behavior_next_states_preds[i])] * (not d)
                else:
                    t[a] = r + self.gamma * np.amax(target_next_state_preds[i]) * (not d)

                inputs[i] = s
                targets[i] = t

            self.behaviour_model.fit(inputs, targets, epochs=1, verbose=0)

    def copy(self):
        """Copies the behavior model's weights into the target model's weights."""

        self.target_model.set_weights(self.behaviour_model.get_weights())

    def save_weights(self):
        """Saves the weights of both models in two h5 files."""

        if not self.save_weights_file_path:
            return
        behavior_save_file_path = re.sub(r'\.h5', r'_beh.h5', self.save_weights_file_path)
        self.behaviour_model.save_weights(behavior_save_file_path)
        target_save_file_path = re.sub(r'\.h5', r'_tar.h5', self.save_weights_file_path)
        self.target_model.save_weights(target_save_file_path)

    def load_weights(self):
        """Loads the weights of both models from two h5 files."""

        if not self.load_weights_file_path:
            return
        behavior_load_file_path = re.sub(r'\.h5', r'_beh.h5', self.load_weights_file_path)
        self.behaviour_model.load_weights(behavior_load_file_path)
        target_load_file_path = re.sub(r'\.h5', r'_tar.h5', self.load_weights_file_path)
        self.target_model.load_weights(target_load_file_path)
