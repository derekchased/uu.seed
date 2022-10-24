import numpy as np
import timeit
from DataAugmentorModel import DataAugmentorModel
from DemonstrationsModel import DemonstrationsModel
from DataAugmentor import DataAugmentor




def demfunc():

    # Get/Create Demonstrations

    dems = {"states":[
    {"global_in":[1, .3, 5, 0, 1],"episode":0},
    {"global_in":[.3, .1, 1, 1, 0],"episode":0}, 
    {"global_in":[3, 4, 2, .5, 0],"episode":0}, 
    {"global_in":[1, .2, 3, 99, 0],"episode":1},
    {"global_in":[.27, 1, 1, .3, 0],"episode":1},
    {"global_in":[2.5, 1.2, 0, 0, 1],"episode":2},
    {"global_in":[1, .4, 99, 2, 1],"episode":3},
    {"global_in":[.5, 1, 0.3, 3, 1],"episode":3}],
    "actions":[1, 0, 1, 1, 0]}

    dem_model = DemonstrationsModel(dems)

    # Define Desired Augmentations
    augmentations = [   DataAugmentorModel(add_state_mixup=True)]#,
                        # DataAugmentorModel(add_noise_gaussian=True), 
                        # DataAugmentorModel(add_noise_uniform=True),
                        # DataAugmentorModel(add_scaling=True),
                        # DataAugmentorModel(add_dropout_continuous=True),
                        # DataAugmentorModel(add_semantic_dropout=True)]
    augmentor = DataAugmentor(dem_model, augmentations)
    augmentor.build_augmentations()

demfunc()
