
from dataclasses import dataclass
from numpy import ndarray
from numpy import asarray
from collections import defaultdict
from typing import Any

@dataclass
class DemonstrationsModel:
	
	augmentation_matrix:Any = None
	demonstrations:dict = None
	states:list = None
	actions:list = None
	states_np:Any = None
	actions_np:Any = None
	num_states:int = 0

	NUM_CONTINOUS:int = 3#28 # first 28 are continuous, after that categorical
	NUM_CATEGORICAL:int = 2#125

	def __post_init__(self):
		if(self.demonstrations is not None):
			self.states = [state['global_in'] for state in self.demonstrations["states"]]
			self.episode_lengths = defaultdict(int)
			for state in self.demonstrations["states"]:
				self.episode_lengths[ state['episode'] ] += 1
			print(f"self.episode_lengths {self.episode_lengths}")
			self.actions = [action for action in self.demonstrations["actions"]]
			self.states_np = asarray(self.states)
			self.actions_np = asarray(self.actions)
			self.num_states = len(self.states)

	def get_all_data(self):
		pass