
from dataclasses import dataclass

@dataclass
class DataAugmentorModel:
	add_noise_gaussian:bool = True
	add_noise_uniform:bool = False
	add_scaling:bool = False
	add_dropout_continuous:bool = False
	add_semantic_dropout:bool = False
	add_state_switch:bool = False
	add_state_mixup:bool = False
	add_adversarial_state_training:bool = False
	num_each_augmentation:int = 3