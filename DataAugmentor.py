
import numpy as np
from DataAugmentorModel import DataAugmentorModel
from DemonstrationsModel import DemonstrationsModel
from typing import Type

class DataAugmentor:
	
	# def __init__(self, demonstrations:Type[DemonstrationsModel], dam:Type[DataAugmentorModel], seed:int = 3):
	def __init__(self, demonstrations:Type[DemonstrationsModel], dams:list, seed:int = 3):
	    self.demonstrations = demonstrations
	    # self.dam = dam
	    self.dams = dams
	    self.default_rng = np.random.default_rng(seed)
	    self.demonstrations = demonstrations


	def build_augmentations(self):
		for dam in dams:
			self.build_augmentation(dam)

	def build_augmentation(self, dam:Type[DataAugmentorModel]):

		dam.num_augmentations



		if dam.add_noise_gaussian:
			pass
		if dam.add_noise_uniform:
			pass
		if dam.add_scaling:
			pass
		if dam.add_dropout_continuous:
			pass
		if dam.add_semantic_dropout:
			pass
		if dam.add_state_switch:
			pass
		if dam.add_state_mixup:
			pass
		if dam.add_adversarial_state_training:
			pass




	def get_gaussian_noise(self):
		pass
    	# return self.default_rng.normal()
    	


	# TODO: ALL NUMBERS ARE HARD-CODED
	# TODO: THIS IS A WORK IN PROGRESS
	def data_augmentation(self, dems, d_type=None, a_type="gauss", with_noise=True):
	    """
	    Data augmentation. The augmentation is not done on the semantic map
	    """
	    scale_gaus = 0.01

	    AUG_DIM = 28 # first 28 are continuous, after that categorical
	    STATE_DIM = 153
	    NUM_AUGMENTATION = 3

	    WITH_SEMANTIC_DROPOUT = False
	    semantic_dropout_rate = 0.3


	    # for each row
	    for i in range(len(dems["states"])):

	        # for each augmentation // do these augmentations
	        for a in range(NUM_AUGMENTATION):
	            
	            # dems is a list:
	            #   dems[states 0][each state   // list    // 0...n][global_in]
	            #   dems[actions 1][each action // integer // 0...n]
	            
	            # extract the specific state, convert to numpy array
	            state = dems["states"][i]
	            aug_state = deepcopy(state["global_in"])
	            aug_state = np.asarray(aug_state)

	            # extract the action
	            aug_action = deepcopy(dems['actions'][i])

	            # (weird)... if no noise, append a clone without any augmentation then continue
	            if not with_noise:
	                dems["states"].append(dict(global_in=aug_state))
	                dems["actions"].append(aug_action)
	                continue

	            # Get noise for just the continuous information, the first AUG_DIM columns
	            noise = np.random.normal(0, scale_gaus, (AUG_DIM))

	            # Concat noise with zeros for the categorical info
	            noise = np.concatenate([noise, np.zeros(STATE_DIM - AUG_DIM)])

	            # dropout a column... currently only applied to categorical
	            if WITH_SEMANTIC_DROPOUT:

	                # randomly select indices (int(125*semantic_dropout_rate)) in range [0,125)... note, hardcoded 125, this is just the size of the categorical info
	                dropout_indexes = np.random.choice(125, int(125*semantic_dropout_rate))
	            
	                # what is this? this suggests that the item at index 27 or 28 is a list or array? not sure how this works with np.asarray from above (which is deprecated without specifying as object type)
	                semantic_map = aug_state[-125:]

	                # dropout values by setting to zero
	                semantic_map[dropout_indexes] = 0.

	            # Be carfeul at the transformer mask value
	            correction = np.ones_like(aug_state)
	            correction[np.where(aug_state == 99)] = 0  # correct noise to 0 if it is masked
	            noise *= correction

	            aug_state += noise

	            dems["states"].append(dict(global_in=aug_state))
	            dems["actions"].append(aug_action)

	            # XXX: to do shuffle

	    return dems



