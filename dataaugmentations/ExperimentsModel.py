from dataclasses import dataclass
from dataaugmentations.DataAugmentorModel import DataAugmentorModel

@dataclass
class ExperimentsModel:
	
	experiments:list = None

	def __post_init__(self):
		if self.experiments is None:
			self.experiments = []
		else:
			experiments_models = []
			for experiment in self.experiments:
				experiments_models.append(DataAugmentorModel(**experiment))
			self.experiments = experiments_models
