

from dataclasses import dataclass
from collections import defaultdict
from typing import Any
import numpy as np
from math import ceil

@dataclass
class DemonstrationsModel:
	
	demonstrations:dict = None
	augmentation_states:Any = None
	augmentation_actions:Any = None
	states:list = None
	actions:list = None
	states_np:Any = None
	actions_np:Any = None
	num_states:int = 0
	ratio: float = 1
	SEED = 3
	SHUFFLE:bool = True
	NUM_CONTINOUS:int = 28 # first 28 are continuous, after that categorical
	NUM_CATEGORICAL:int = 125

	def __post_init__(self):

		if(self.demonstrations is not None):
			self.states = [state['global_in'] for state in self.demonstrations["states"]]
			self.actions = [action for action in self.demonstrations["actions"]]
			self.episode_lengths = defaultdict(int)
			self.states_np = np.asarray(self.states)
			self.actions_np = np.asarray(self.actions)

			for state in self.demonstrations["states"]:
				self.episode_lengths[ state['episode'] ] += 1

			if self.ratio < 1:
				# round to nearest even number (to keep classes balanced [during demonstration, alt betw two trajects])
				num_episodes = ceil( len(self.episode_lengths) * self.ratio / 2) * 2
				episodeindices = list(self.episode_lengths.values())
				cumsum = np.cumsum(episodeindices)
				self.demonstrations["states"] = self.demonstrations["states"][:cumsum[num_episodes-1]]
				self.demonstrations["actions"] = self.demonstrations["actions"][:cumsum[num_episodes - 1]]
				self.states = [state['global_in'] for state in self.demonstrations["states"]]
				self.actions = [action for action in self.demonstrations["actions"]]
				self.episode_lengths = defaultdict(int)
				self.states_np = np.asarray(self.states)
				self.actions_np = np.asarray(self.actions)
				for state in self.demonstrations["states"]:
					self.episode_lengths[state['episode']] += 1
				episodeindices = list(self.episode_lengths.values())
				cumsum = np.cumsum(episodeindices)

			self.num_states = len(self.states)

	def get_all_states(self):
		return np.concatenate( (self.states_np, self.augmentation_states) )

	def get_all_actions(self):
		return np.concatenate( (self.actions_np, self.augmentation_actions) )

	def get_kiwi_demonstrations_with_augmentations(self):

		# Get all states and corresponding actions (original and augmented)
		all_states = self.get_all_states()
		all_actions = self.get_all_actions()

		if self.SHUFFLE:
			# shuffle indices, states, actions
			indices = np.random.default_rng(self.SEED).permutation(len(all_states))
			all_states = all_states[indices]
			all_actions = all_actions[indices]
		else:
			indices = np.arange(len(all_states))

		# create kiwi compatible data structure for training
		kiwi_dems = {"states": [{"global_in": np.asarray(state)} for state in all_states], "actions":list(all_actions)}
		return kiwi_dems, all_states, all_actions, indices
		