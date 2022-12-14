import time
import tensorflow as tf
import os
from ruamel.yaml import YAML
from itertools import combinations
from architectures.kiwi_arch import *
from agents.ppo_agent import PPOAgent
from reward_model.demonstrations_loader import DemonstrationLoader
from reward_model.behavioral_cloning import BehavioralCloning
from reward_model.demonstrations_loader import DemonstrationLoader

from dataaugmentations.TrainingModel import TrainingModel
from dataaugmentations.ExperimentsModel import ExperimentsModel
from dataaugmentations.DemonstrationsModel import DemonstrationsModel
from dataaugmentations.DataAugmentor import DataAugmentor
from dataaugmentations.DataAugmentorModel import DataAugmentorModel
from pathlib import Path

from datetime import datetime
import copy

class BatchTrainer:
    """
	1. receive train and expermiment model from shell
	2. load demonstrations
	3. create agent
	4. for each experiment:
		a. create augmentation data set
		b. train the agent
		c. save the model
	"""

    def __init__(self, train_config: dict, experiment_list: list, meta:dict):

        # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        # tf.config.list_physical_devices('GPU')
        # tf.debugging.set_log_device_placement(True)
        #
        # # Tensorflow GPU stuff
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # if len(physical_devices) > 0:
        #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # convert configs to model
        self.train_model = TrainingModel(**train_config)
        self.exp_model = ExperimentsModel(experiment_list)
        self.meta = meta

        self.yml = YAML(pure=True)
        self.yml.preserve_quotes = True
        self.yml.boolean_representation = ['False', 'True']

        self.filename = f"{self.meta['ID']}_train.yml"
        path = Path(self.filename, missing_ok = True)
        try:
            path.unlink()
        except:
            pass

        print(self.train_model)
        print(self.exp_model)

        # if no experiments in the list, and max combo is set, and num augs is set (this is done in kiwi.yml)
        # We now treat this is a combinatoric instead of a manually set list of augs
        if not len(self.exp_model.experiments) and self.train_model.MAX_COMBINATIONS is not None and self.train_model.NUM_AUGS is not None:
            # create list to hold DataAugmentorModel's
            model_list = []

            # Get first DA object, name it base, this will have no augmentations
            model = DataAugmentorModel(name="base")
            model_list.append(model)

            # Get the second DA object, name it basenrml, this will have no augmentations, but will extend it's
            # data size to match that of the augmentations
            model = DataAugmentorModel(name="base", num_augs=self.train_model.NUM_AUGS)
            model_list.append(model)

            # copy the labels from the model for use in generating models
            # hack... this should be static or something
            aug_labels = copy.copy(model.augmentation_obj)

            # iterate through each size of combinations
            for size in range(1, self.train_model.MAX_COMBINATIONS + 1):

                # generate all combinations of size
                aug_combinations = combinations(aug_labels, size)

                # iterate over each combo of size
                for aug_combo in aug_combinations:

                    # create a dict which can be used to create DataAugmentorModel from the combination
                    augs = {aug_label: True for aug_label in aug_combo}

                    # skip these models because they don't make sense
                    if "add_noise_gaussian" in augs and "add_noise_uniform" in augs:
                        continue

                    # add num_augs to the dict
                    augs["num_augs"] = self.train_model.NUM_AUGS

                    # create model from the dict
                    model = DataAugmentorModel(**augs)

                    # add to list
                    model_list.append(model)

            # update list of experiments
            self.exp_model.experiments = model_list

        # for experiment in self.exp_model.experiments:
        #     print(f"{experiment.get_name()}\n{experiment.augmentation_obj}\n")


        # Create additional experiments based on the list of demonstration_ratios
        if len(self.train_model.demonstration_ratios)>1:
            ratio_experiments = []
            for ratio in self.train_model.demonstration_ratios:
                if ratio == 1:
                    continue
                else:
                    for experiment in self.exp_model.experiments:
                        ratio_exp = copy.copy(experiment)
                        ratio_exp.ratio = ratio
                        ratio_experiments.append(ratio_exp)
            self.exp_model.experiments = self.exp_model.experiments + ratio_experiments

        # Create additional experiments based on the list of gaussian scales
        if len(self.train_model.gaussian_scale)>1:
            gaus_experiments = []
            for gaus in self.train_model.gaussian_scale:
                for experiment in self.exp_model.experiments:
                    if not experiment.add_noise_gaussian:
                        continue
                    elif gaus == experiment.GAUSSIAN_SCALE:
                        continue
                    else:
                        gaus_exp = copy.copy(experiment)
                        gaus_exp.GAUSSIAN_SCALE = gaus
                        label = str(gaus).split("0")
                        gaus_experiments.append(gaus_exp)

            self.exp_model.experiments = self.exp_model.experiments + gaus_experiments

        for index, experiment in enumerate(self.exp_model.experiments):
            print(f"{index}. {experiment.get_name()}\n{experiment.augmentation_obj}\n")

        # path to folder where models will be stored
        self.MODEL_PATH = os.path.join(self.train_model.PATH, self.train_model.MODEL_PATH)

        # path to txt files
        self.DEMONSTRATIONS_PATH = os.path.join(self.train_model.PATH, self.train_model.DEMONSTRATIONS_PATH,
                                                self.train_model.DEMONSTRATIONS)
        self.TRAINER_PATH = os.path.join(self.train_model.PATH, self.train_model.TRAINER_PATH, self.train_model.TRAINER)
        self.STATISTICS_PATH = os.path.join(self.train_model.PATH, self.train_model.STATISTICS_PATH,
                                            self.train_model.STATISTICS)

        # Instantiate the demonstrations loader


        # types of agents to train
        if self.train_model.TRAINING_AGENTS == 1:
            self.agents_type = ['normal']
        elif self.train_model.TRAINING_AGENTS == 2:
            self.agents_type = ['normal', 'impro']
        elif self.train_model.TRAINING_AGENTS == 3:
            self.agents_type = ['normal', 'impro', 'strict']
        else:
            self.agents_type = ['normal']

        self.batch_train()

    def batch_train(self):

        # timestamp
        experiment_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

        self.train_model.timestamp = experiment_timestamp

        self.train_model.training_stats = []

        start = time.time()

        for agent_type in self.agents_type:

            for experiment in self.exp_model.experiments:

                for combo in self.train_model.MAX_COMBINATIONS:


                    self.dem_loader = DemonstrationLoader(self.TRAINER_PATH, self.STATISTICS_PATH)

                    demonstrations = self.dem_loader.load_demonstrations_txt([self.DEMONSTRATIONS_PATH],
                                                                             agent_type,
                                                                             actions_type=self.train_model.ACTION_TYPE)

                    agent = self.get_agent(agent_type, experiment.name)

                    dem_model = self.get_augmentations(demonstrations, experiment)

                    print(f"before - {dem_model.states_np.shape} - after {dem_model.get_all_states().shape}")

                    kiwi_dems, all_states, all_actions, indices = dem_model.get_kiwi_demonstrations_with_augmentations()

                    model_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

                    stats = self.train(agent, agent_type, kiwi_dems, experiment.name, model_timestamp)

                    stats.update({"before":f"{dem_model.states_np.shape}", "after":f"{dem_model.get_all_states().shape}"})

                    self.write_stats({stats["name"]: stats})

                    self.train_model.training_stats.append(stats)

        end = time.time()

        self.finalize_stats(experiment_timestamp, int(end-start))

        print("Total time: {}s, {}m, {}h".format(end - start, (end - start) / 60, (end - start) / (60 * 60)))

    def write_stats(self, stats):
        with open(self.filename, 'a') as f:
            documents = self.yml.dump(stats, f)

    def finalize_stats(self, timestamp, total):


        # {stats["name"]: stats}
        self.meta.update( {"timestamp":timestamp, "seconds":total} )

        with open(self.filename, 'r') as f:

            stats = self.yml.load(f)

        kiwi = {"meta":self.meta, "stats":stats}

        final_yml = {"kiwi":kiwi}

        with open(self.filename, 'w') as f:
            documents = self.yml.dump(final_yml, f)


    def get_augmentations(self, demonstrations, experiment):
        ratio = experiment.ratio
        dem_model = DemonstrationsModel(demonstrations=demonstrations, ratio=ratio)
        augmentor = DataAugmentor(dem_model, [experiment])
        augmentor.build_augmentations()
        return dem_model

    def get_agent(self, agent_type, model_name):
        # We will have 3 agents: normal, impro and strict
        graph = tf.compat.v1.Graph()

        with graph.as_default():
            tf.compat.v1.disable_eager_execution()
            sess = tf.compat.v1.Session(graph=graph)

            # These are PPO agents because they can be easily use with my GAIL implementation. Probably we need to
            # change the PPO class name, though.
            agent = PPOAgent(sess, input_spec=input_spec, network_spec=network_spec, obs_to_state=obs_to_state,
                             action_type=self.train_model.ACTION_TYPE, action_size=self.train_model.ACTION_SIZE,
                             name=f"{model_name}_{agent_type}", model_name=f"{model_name}_{agent_type}")

            # Modify the agent with Behavioral Cloning module
            agent.bc_module = BehavioralCloning(agent, self.dem_loader, bc_lr=0.0005)
            # Initialize variables of models
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        return agent

    def train(self, agent, agent_type, demonstrations, model_name, timestamp):
        try:
            start = time.time()

            agent.bc_module.set_dataset(demonstrations, with_validation=False)

            training_stats = agent.bc_module.train(num_itr=self.train_model.epochs, with_validation=False,
                                                   save_statistics=True, statistics_path=self.STATISTICS_PATH)

            agent.save_model(f"{timestamp}_{model_name}_{agent_type}", onnx_path=self.MODEL_PATH)

            print(f"{model_name}_{agent_type} saved to {self.MODEL_PATH}")

            self.dem_loader.close_training()

            end = time.time()

            seconds = end - start
            minutes = (end - start) / 60
            hours = (end - start) / (60 * 60)

            loss = [float(training_stat["loss"]) for training_stat in training_stats]
            training_info = {'loss': loss, "name": f"{model_name}_{agent_type}", "train_time": seconds,
                             "timestamp": timestamp}

            print(f"Training time: {seconds}s, {minutes}m, {hours}h")

            return training_info

        except Exception as e:
            print("\n\n====== async_train EXCEPTION")
            print(e)
