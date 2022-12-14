import yaml

class YMLLoader:
	
	@staticmethod
	def parse_yml(YML_PATH):
		
		# open yml file safely
		with open(YML_PATH,"r") as yml_file:
			yml = yaml.safe_load(yml_file)

		# get training config from yml
		yml_training = yml['kiwi']['defaults']['training']
		
		# get default augmentation num augs from yml
		default_num_augs = yml['kiwi']["defaults"]['experiments']["num_augs"]

		# create list to hold experiments
		experiments = []

		meta = yml['kiwi']['meta']

		# get list of experiments from yml
		if "experiments" in yml['kiwi']:
			yml_experiments = yml['kiwi']['experiments']

			# loop through experiments in yml and convert to list of dataugmentation dicts
			for experiment_key, experiment in zip(yml_experiments.keys(), yml_experiments.values()):
				print(experiment_key, experiment)
				# use the experiment key as the name of this aug
				aug_dict = {"name":experiment_key}

				# set num augs, using the default from above if it is not specifically set
				aug_dict["num_augs"] = default_num_augs if not "num_augs" in experiment else experiment["num_augs"]

				# extract augmentation combination list
				if "augmentations" in experiment:
					aug_keys = experiment["augmentations"]
					aug_dict.update({key: True for key in aug_keys})
					#
					# aug_dict[key] = True for key in aug_keys
					# for key in aug_keys: aug_dict[key] = True

					# convert list to dict where value of each augmentation is True
					# aug_dict = {key: True for key in aug_keys}

				if "max" in experiment:
					aug_dict["max"] = experiment["max"]

				print(aug_dict)
				# add to the list of experiments
				experiments.append(aug_dict)

		return yml_training, experiments, meta

	@staticmethod
	def parse_experiment_yml(YML_PATH):
		experiment_yml = YMLLoader.open_yml_file(YML_PATH)

		return experiment_yml

	@staticmethod
	def open_yml_file(YML_PATH):
		# open yml file safely
		with open(YML_PATH, "r") as yml_file:
			yml = yaml.safe_load(yml_file)
		return yml