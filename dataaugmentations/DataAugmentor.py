
import numpy as np
from dataaugmentations import DataAugmentorModel
from dataaugmentations import DemonstrationsModel

class DataAugmentor:
	
	def __init__(self, demonstrations_model, augmentations_model, seed = 3):
		# demonstrations_model holds the original demonstrations.txt object, 
		# numpy version of states and actions, and will hold the augmentation 
		# data which  can be concatenated with the original data to have a
		# full data set
	    self.dems_model = demonstrations_model

	    # augmentations_model is a list of augmentation objects. In each object
	    # can specify a combination of augmentations to perform at once, and
	    # how many times per state to do it. 
	    # 
	    self.augs_model = augmentations_model
	    self.rng = np.random.default_rng(seed)

	def build_augmentations(self):
		"""
	    Create one numpy array containing all augmentations.

	    Note- this function creates one large numpy array and then uses
	    array slicing to act on different parts of the array. 
	    Removed as many for loops as possible in favor of creating larger 
	    numpy arrays, then iterating through them with slices, rather than
	    generating smaller numpy arrays. For example, if 3 augmentations are 
	    created, we generate one large numpy array of gaussian noise then 
	    iterate through it in 3 sections rather than generate 3 different
	    gaussian arrays.

	    Parameters:
			
	    Returns:

	    """

		# get shape of demonstrations
		num_states, num_features = self.dems_model.states_np.shape

		# get total number of augmentations
		total_augmentations = sum([aug_model.num_augs 
			for aug_model in self.augs_model])

		# copy the full set of demonstration states total_augmentations number
		# of times and vertically stack in a single numpy array
		augmentation_states = np.resize( 
			self.dems_model.states_np, 
			(num_states*total_augmentations, num_features) )

		# do the same for the actions
		self.dems_model.augmentation_actions = np.resize( 
			self.dems_model.actions_np, 
			num_states*total_augmentations)

		# STORE INDICES OF TRANSFORMER elements to set back to 99 later
		indices_transformer = augmentation_states == 99
		
		# keep track of vertical index for iterating through the different sections of
		# the numpy array
		curr_index = 0 

		# iterate through each requested augmentation model and act on the 
		# numpy array
		for ind, aug_model in enumerate(self.augs_model):

			# calculate the slice of the matrix to operate on
			end_index = curr_index + (num_states*aug_model.num_augs)

			# select the slice of the full matrix for this augmentation
			augmentation_slice = augmentation_states[curr_index:end_index,:]

			# select the slice of the full matrix for this augmentation
			self.build_kiwi_augmentation(aug_model, augmentation_slice)
			
			# update index for next increment
			curr_index = end_index

		# set transformer elements back to 99
		augmentation_states[indices_transformer] = 99

		# save the augmentation array to the model
		self.dems_model.augmentation_states = augmentation_states



	def build_kiwi_augmentation(self, aug_model, aug_slice):
		"""
	    Performs the indicated augmentations onto the input data. Takes into
	    account the current kiwi structure which is loosely documented, such
	    as number of continuous and categorical variables and the use of 99
	    for marking the transformer elements.

	    Note - This function works on numpy views (slices).
	    Note - The augmentations will "stack" and it may not make sense.

        Parameters:
			aug_model(DataAugmentorModel):				
				Contains information on which augmentations to perform and
				stores the augmentation array for augmentations which that
				makes sense.
			aug_slice(ndarray):			
				a view of a numpy array containing some multiple of the 
				original demonstration input

        Returns:
        	None: Acts directly on the aug_slice
	    """

	    # store some variables locally for easier code reading
		NUM_CONTINOUS = self.dems_model.NUM_CONTINOUS
		num_states_dems = self.dems_model.states_np.shape[0]

		# get shape of slice to be used in augmentations
		num_states_augs, num_features = aug_slice.shape

		# store indices where element is 99, set these back to 99 after each 
		# augmentation
		indices_transformer_slice = aug_slice == 99

		if aug_model.add_noise_gaussian:
			
			# generate gaussian noise for continuous features only
			augmentation_obj = self.rng.normal(loc=0.0, 
				scale=aug_model.GAUSSIAN_SCALE, 
				size=(num_states_augs, NUM_CONTINOUS))

			# store indices where element is 99, set these to 0 for no effect
			indices_transformer = aug_slice[:, 0:NUM_CONTINOUS] == 99
			augmentation_obj[indices_transformer] = 0

			# add noise to the augmentation array
			aug_slice[:,0:NUM_CONTINOUS] += augmentation_obj
			

		if aug_model.add_noise_uniform:
			
			# generate uniform noise for continuous features only
			augmentation_obj = self.rng.uniform(low=aug_model.UNIFORM_LOW, 
				high=aug_model.UNIFORM_HIGH, size=(num_states_augs, NUM_CONTINOUS))

			# store indices where element is 99, set these to 0 for no effect
			indices_transformer = aug_slice[:,0:NUM_CONTINOUS] == 99
			augmentation_obj[indices_transformer] = 0

			# add noise to the augmentation array
			aug_slice[:,0:NUM_CONTINOUS] += augmentation_obj

		if aug_model.add_scaling:
			
			# generate scaling amplitude for continuous features only
			augmentation_obj = self.rng.uniform(low=aug_model.RAS_LOW, 
				high=aug_model.RAS_HIGH, size=(num_states_augs, NUM_CONTINOUS))

			# store indices where element is 99, set these to 1 for no effect
			indices_transformer = aug_slice[:,0:NUM_CONTINOUS] == 99
			augmentation_obj[indices_transformer] = 1

			# multiply the array by the scaling factor
			aug_slice[:,0:NUM_CONTINOUS] *= augmentation_obj

		if aug_model.add_state_switch:
			#print(f"\nstate_switch\n")
			pass

		if aug_model.add_state_mixup:


			# Get length of each episode then convert into a cumulative 
			# sum. This represents the first row of each new episode. 
			# Needed so we don't add the last state of an episode to the 
			# first state of the next episode
			episodeindices = list(self.dems_model.episode_lengths.values())
			cumsum = np.cumsum(episodeindices)
			
			# Create augmentation array where the number of rows is equal
			# to the number of original demonstrations, and the
			# number of features is for continuous features only
			augmentation_obj = np.zeros( 
				(num_states_dems, NUM_CONTINOUS))

			# Copy the input into the augmentation_obj but clip first row. 
			# Note, the last row will be all zeros
			augmentation_obj[0:-1,:] = aug_slice[1:num_states_dems,:NUM_CONTINOUS]
			
			# Set the first row of each episode to zero. Subtract one because
			# clipped the first row above.
			augmentation_obj[cumsum-1,:] = 0

			# Copy and extend the array veritcally for each num_augmentations
			# now we have repeating array of the original demonstrations where
			# the first state of the first episode has been removed and replaced
			# with last row in the slice set to all zeros
			augmentation_obj = np.resize(augmentation_obj, 
				(len(augmentation_obj)*aug_model.num_augs, NUM_CONTINOUS))

			# generate the beta distribution for the size of the aug array
			beta_distr = self.rng.beta(aug_model.BETA_1, 
				aug_model.BETA_2, size=augmentation_obj.shape)

			# store indices where element is 99, set these back to 99 after
			# the operation
			indices_transformer = aug_slice == 99

			# do state mixup
			aug_slice[:,0:NUM_CONTINOUS] = beta_distr*aug_slice[:,0:NUM_CONTINOUS] + (1-beta_distr)*augmentation_obj

			# set elements back to 99
			aug_slice[indices_transformer] = 99

		if aug_model.add_adversarial_state_training:
			pass

		if aug_model.add_dropout_continuous:
			
			# generate dropout indices for continuous features (columns 0 through 27)
			dropout_continuous_indices = self.rng.choice(a=NUM_CONTINOUS, 
				size=(num_states_augs, 
					int(NUM_CONTINOUS*aug_model.DROPOUT_RATE_CONTINUOUS)))
			
			# dropout indices from each state by setting to 0
			aug_slice[np.arange(num_states_augs),dropout_continuous_indices.T] = 0 

			# for debugging
			augmentation_obj = (np.arange(num_states_augs),dropout_continuous_indices.T)

		if aug_model.add_semantic_dropout:

			# generate dropout indices for semantic features (columns 28 through 153)
			dropout_semantic_indices = self.rng.choice(
				a=np.arange(NUM_CONTINOUS,num_features), 
				size=(num_states_augs,
					int(self.dems_model.NUM_CATEGORICAL*aug_model.DROPOUT_RATE_CATEGORICAL)))

			# store indices where element is 99, set these back to 99 after
			# the operation
			indices_transformer = aug_slice == 99

			# dropout indices from each state by setting to 0
			aug_slice[np.arange(num_states_augs),dropout_semantic_indices.T] = 0 

			# set elements back to 99
			aug_slice[indices_transformer] = 99

			# for debugging
			augmentation_obj = (np.arange(num_states_augs),dropout_semantic_indices.T)

		# for debugging
		aug_model.augmentation_obj = augmentation_obj		