
from dataclasses import dataclass

@dataclass
class DataAugmentorModel:
	add_noise_gaussian:bool = False
	add_noise_uniform:bool = False
	add_scaling:bool = False
	add_dropout_continuous:bool = False
	add_semantic_dropout:bool = False
	add_state_switch:bool = False
	add_state_mixup:bool = False
	add_adversarial_state_training:bool = False
	num_augmentations:int = 2

	# ALPHA, BETA
	BETA_1 = .4
	BETA_2 = .4
	GAUSSIAN_SCALE = 0.0003
	UNIFORM_LOW = -0.0003
	UNIFORM_HIGH = 0.0003
	RAS_LOW = 0.0003
	RAS_HIGH = 0.0006
	DROPOUT_RATE_CONTINUOUS = 1 # this is experimental, (.1,.3)
	DROPOUT_RATE_CATEGORICAL = .8 # this is experimental, (.1,.3)