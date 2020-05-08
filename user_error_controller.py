
import random
from dialogue_config import usersim_intents


class UserErrorController:

    def __init__(self, dict_db, dict_const):

        self.intents = usersim_intents
        self.prob_error_of_intent = dict_const['emc']['intent_error_prob']
        self.mode_error_of_slot = dict_const['emc']['slot_error_mode']  
        self.prob_error_of_slot = dict_const['emc']['slot_error_prob']
        self.dict_movie = dict_db
        
    def new_slot(self, key, dict_inform):

        dict_inform.pop(key)
        movie_keys = [key for key in self.dict_movie.keys()]
        rnd_slt = random.choice(movie_keys)
        dict_inform[rnd_slt] = random.choice(self.dict_movie[rnd_slt])

    def remove_slot(self, key, dict_inform):

        dict_inform.pop(key)
        
    def noise_value_slot(self, key, dict_inform):

        dict_inform[key] = random.choice(self.dict_movie[key])

        
    def add_user_action_error(self, action):

        dict_inform = action['inform_slots']
        action_keys = [key for key in action['inform_slots'].keys()]
        for key in action_keys:
            assert key in self.dict_movie
            if random.random() < self.prob_error_of_slot:
                if self.mode_error_of_slot == 0:  # replace the slot_value only
                    self.noise_value_slot(key, dict_inform)
                elif self.mode_error_of_slot == 1:  # replace slot and its values
                    self.new_slot(key, dict_inform)
                elif self.mode_error_of_slot == 2:  # delete the slot
                    self.remove_slot(key, dict_inform)
                else:  # Combine all three
                    rand_choice = random.random()
                    if rand_choice <= 0.33:
                        self.noise_value_slot(key, dict_inform)
                    elif rand_choice > 0.33 and rand_choice <= 0.66:
                        self.new_slot(key, dict_inform)
                    else:
                        self.remove_slot(key, dict_inform)
        if random.random() < self.prob_error_of_intent:  # add noise for intent level
            action['intent'] = random.choice(self.intents)
