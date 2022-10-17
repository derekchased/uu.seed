
from dataclasses import dataclass
# from typing import Any
from typing import Type
from numpy import ndarray
from numpy import asarray

@dataclass
class DemonstrationsModel:
	
	kiwi_demonstrations:dict = None
	states:list = None
	actions:list = None
	states_np:Type[ndarray] = None
	actions_np:Type[ndarray] = None
	NUM_CONTINOUS:int = 28 # first 28 are continuous, after that categorical
	NUM_CATEGORICAL:int = 125

	def __post_init__(self):
		if(self.kiwi_demonstrations is not None):
			self.states = [state['global_in'] for state in self.kiwi_demonstrations["states"]]
			self.actions = [action for action in self.kiwi_demonstrations["actions"]]
			self.states_np = asarray(self.states)
			self.actions_np = asarray(self.actions)