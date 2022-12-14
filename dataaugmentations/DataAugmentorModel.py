from typing import Any
from dataclasses import dataclass

@dataclass
class DataAugmentorModel:
	
	name:str = None
	index:int = 1
	add_noise_gaussian:bool = False
	add_noise_uniform:bool = False
	add_scaling:bool = False
	add_dropout_continuous:bool = False
	add_semantic_dropout:bool = False
	# add_state_switch:bool = False
	add_state_mixup:bool = False
	add_adversarial_state_training:bool = False

	augmentation_obj = {"add_noise_gaussian": False,
						"add_noise_uniform": False,
						"add_scaling": False,
						"add_dropout_continuous": False,
						"add_semantic_dropout": False,
						# "add_state_switch": False,
						"add_state_mixup": False,
						# "add_adversarial_state_training": False
						}

	augmentation_labels = {"add_noise_gaussian": "gaus",
						"add_noise_uniform": "uni",
						"add_scaling": "sca",
						"add_dropout_continuous": "drc",
						"add_semantic_dropout": "drs",
						# "add_state_switch": "ss",
						"add_state_mixup": "sm",
						# "add_adversarial_state_training": "ast"
						}


	num_augs:int = 1
	ratio:float = 1
	aug_dict:dict = None

	# debugging
	augmentation_obj:Any = None

	# ALPHA, BETA
	BETA_1 = .4
	BETA_2 = .4
	GAUSSIAN_SCALE = 0.0003
	UNIFORM_LOW = -0.0003
	UNIFORM_HIGH = 0.0003
	RAS_LOW = 0.0003
	RAS_HIGH = 0.0006
	DROPOUT_RATE_CONTINUOUS = .1
	DROPOUT_RATE_CATEGORICAL = .1

	def get_name(self):
		labels = [ele for ele in [self.augmentation_labels[key] if val else "" for key, val in self.augmentation_obj.items()] if not ele == ""]
		if len(labels):
			name = "_".join(labels)
		else:
			name = self.name
		name = name + f"_r{int(self.ratio*100)}"
		if self.add_noise_gaussian:
			label = str(self.GAUSSIAN_SCALE).split("0")
			name = name + f"_g{label[-1]}_e{len(label)}"
		name = name + f"_m{self.index}"
		return name

	def __post_init__(self):
		if(self.augmentation_obj is None):
			self.augmentation_obj = {"add_noise_gaussian":self.add_noise_gaussian,
									 "add_noise_uniform":self.add_noise_uniform,
									 "add_scaling": self.add_scaling,
									 "add_dropout_continuous": self.add_dropout_continuous,
									 "add_semantic_dropout": self.add_semantic_dropout,
									 # "add_state_switch": self.add_state_switch,
									 "add_state_mixup": self.add_state_mixup,
									 # "add_adversarial_state_training": self.add_adversarial_state_training}
									 }