
from dataclasses import dataclass
# from numpy import asarray
# from numpy import concatenate
from collections import defaultdict
from typing import Any
import numpy as np

@dataclass
class DemonstrationsModel:
	
	augmentation_states:Any = None
	augmentation_actions:Any = None
	demonstrations:dict = None
	states:list = None
	actions:list = None
	states_np:Any = None
	actions_np:Any = None
	num_states:int = 0

	NUM_CONTINOUS:int = 28 # first 28 are continuous, after that categorical
	NUM_CATEGORICAL:int = 125

	def __post_init__(self):
		if(self.demonstrations is not None):
			self.states = [state['global_in'] for state in self.demonstrations["states"]]
			self.episode_lengths = defaultdict(int)
			for state in self.demonstrations["states"]:
				if(hasattr(state,'episode')):
					print("yes")
					self.episode_lengths[ state['episode'] ] += 1
				else:
					print("no")
			# print(f"self.episode_lengths {self.episode_lengths}")
			self.actions = [action for action in self.demonstrations["actions"]]
			self.states_np = np.asarray(self.states)
			self.actions_np = np.asarray(self.actions)
			self.num_states = len(self.states)

	def get_all_states(self):
		return np.concatenate( (self.states_np, self.augmentation_states) )

	def get_all_actions(self):
		return np.concatenate( (self.actions_np, self.augmentation_actions) )

	def get_kiwi_demonstrations_with_augmentations(self):
		print("get_kiwi_demonstrations_with_augmentations")
		all_states = self.get_all_states()
		all_actions = self.get_all_actions()
		indices = np.random.default_rng().shuffle(np.arange(len(all_states)))
		all_states = all_states[indices]
		all_actions = all_actions[indices]
		kiwi_dems = {"states": [{"global_in": np.asarray(state)} for state in all_states], "actions":list(all_actions)}
		print("kiwi_dems")
		print(kiwi_dems)
		return kiwi_dems
		