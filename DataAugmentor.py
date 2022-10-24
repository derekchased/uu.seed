
import numpy as np
from DataAugmentorModel import DataAugmentorModel
from DemonstrationsModel import DemonstrationsModel
from typing import Type

class DataAugmentor:
	
	def __init__(self, demonstrations_model:Type[DemonstrationsModel], augmentation_models:list, seed:int = 3):
	    self.demonstrations_model = demonstrations_model
	    self.augmentation_models = augmentation_models
	    self.default_rng = np.random.default_rng()#seed)

	def build_augmentations(self):

		# 1. build full numpy matrix

		# get shape of demonstrations
		num_states, num_features = self.demonstrations_model.states_np.shape

		# get total number of augmentations
		total_augmentations = sum([aug_model.num_augmentations for aug_model in self.augmentation_models])

		# create an augmentation matrix by repeating the full demonstrations for each augmentation, using np.resize()
		self.aug_matrix = np.resize( self.demonstrations_model.states_np, (num_states*total_augmentations, num_features) )

		# print(f"demonstrations\n{self.demonstrations_model}")

		print(f"\nself.aug_matrix\n{self.aug_matrix}")

		# XXX TO DO - STORE INDICES OF TRANSFORMER, 99?


		
		# print(f"num_states {num_states}, num_features {num_features}, total_augmentations {total_augmentations}")
		# print("\n",self.demonstrations_model.states_np,"\n",self.aug_matrix)

		# 2. iterate through each augmentation

		# keep track of index
		curr_index = 0 
		for ind, aug_model in enumerate(self.augmentation_models):
			# print("\n\n",ind, aug_model)

			# calculate the slice of the matrix to operate on
			end_index = curr_index + (num_states*aug_model.num_augmentations)

			# select the slice of the full matrix for this augmentation
			augmentation_slice = self.aug_matrix[curr_index:end_index,:]

			# select the slice of the full matrix for this augmentation
			self.build_kiwi_augmentation(aug_model, augmentation_slice)
			
			# update index for next increment
			curr_index = end_index

		print(f"\nself.aug_matrix after\n{self.aug_matrix}")


	def build_kiwi_augmentation(self, aug_model:Type[DataAugmentorModel], aug_slice):

		# get shape of slice to be used in augmentations
		num_states, num_features = aug_slice.shape

		# kiwi: the first X columns are continuous, the remaining are categorical
		
		# print(aug_slice)

		if aug_model.add_noise_gaussian:
			# generate noise
			
			gaus_noise = self.default_rng.normal(loc=0.0, scale=aug_model.GAUSSIAN_SCALE, size=(num_states, self.demonstrations_model.NUM_CONTINOUS))
			print(f"\ngaus_noise\n{gaus_noise}")

			# add noise to the array
			aug_slice[:,0:self.demonstrations_model.NUM_CONTINOUS] += gaus_noise
			
		if aug_model.add_noise_uniform:
			# generate noise 
			
			uniform_noise = self.default_rng.uniform(low=aug_model.UNIFORM_LOW, high=aug_model.UNIFORM_HIGH, size=(num_states, self.demonstrations_model.NUM_CONTINOUS))

			print(f"\nuniform_noise\n{uniform_noise}")

			# add noise to the array
			aug_slice[:,0:self.demonstrations_model.NUM_CONTINOUS] += uniform_noise

		if aug_model.add_scaling:
			
			# generate scaling amplitude 
			ras_noise = self.default_rng.uniform(low=aug_model.RAS_LOW, high=aug_model.RAS_HIGH, size=(num_states, self.demonstrations_model.NUM_CONTINOUS))

			print(f"\nras_noise\n{ras_noise}")

			# add noise to the array
			aug_slice[:,0:self.demonstrations_model.NUM_CONTINOUS] *= ras_noise
			
		if aug_model.add_state_switch:
			print(f"\nstate_switch\n")
			pass

		if aug_model.add_state_mixup:
			print(f"\nstate_mixup\n")
			episodeindices = list(self.demonstrations_model.episode_lengths.values())
			cumsum = np.cumsum(episodeindices) - 1
			mixup = np.zeros( (self.demonstrations_model.num_states,self.demonstrations_model.NUM_CONTINOUS) )
			mixup[0:-1,:] = aug_slice[1:self.demonstrations_model.num_states,:self.demonstrations_model.NUM_CONTINOUS]
			mixup[cumsum,:] = 0
			mixup = np.resize(mixup, (len(mixup)*aug_model.num_augmentations,mixup.shape[1]))
			print("mixup\n",mixup)
			aug_slice[:,0:self.demonstrations_model.NUM_CONTINOUS] += mixup


		if aug_model.add_adversarial_state_training:
			print(f"\nadversarial\n")
			pass

		if aug_model.add_dropout_continuous:
			
			print("aug_model.add_dropout_continuous")
			dropout_continuous_indices = self.default_rng.choice(a=self.demonstrations_model.NUM_CONTINOUS, 
				size=(num_states, int(self.demonstrations_model.NUM_CONTINOUS*aug_model.DROPOUT_RATE_CONTINUOUS)))
			
			print(f"\ndropout_continuous.T\n{dropout_continuous_indices.T}")

			# aug_slice[:,dropout_continuous_indices] = 0 
			aug_slice[np.arange(num_states),dropout_continuous_indices.T] = 0 


		if aug_model.add_semantic_dropout:
			print("aug_model.add_semantic_dropout")
			
			# generate dropout indices for continuous features
			dropout_semantic_indices = self.default_rng.choice(a=np.arange(self.demonstrations_model.NUM_CONTINOUS,num_features), 
				size=(num_states,int(self.demonstrations_model.NUM_CATEGORICAL*aug_model.DROPOUT_RATE_CATEGORICAL)))

			print(f"\ndropout_semantic_indices.T\n{dropout_semantic_indices.T}")

			aug_slice[np.arange(num_states),dropout_semantic_indices.T] = 0 


