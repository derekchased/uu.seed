
import argparse
from dataaugmentations.YMLLoader import YMLLoader
from dataaugmentations.BatchTrainer import BatchTrainer

class Shell:
	
	"""
	1. get arguments
	2. load config
	3. inject
	"""

	def __init__(self):

		# create parser
		parser = argparse.ArgumentParser()

		# add arguments
		parser.add_argument('-y', '--yml', help="Config file in yml format", default="./experiments/kiwi.yml", type=str)
		parser.add_argument('-at', '--action-type', choices=['discrete', 'continuous'], 
			help="The type of action space [discrete, continuous]", 
			dest="ACTION_TYPE")
		parser.add_argument('-as', '--action-size', help="The size of action space", 
			type=int, dest="ACTION_SIZE")
		parser.add_argument('-ta', '--training-agents', choices=[1, 2, 3], type=int,
			help="The number of agents you want to train from [normal, impro, strict]", 
			dest="TRAINING_AGENTS")	

		# parse 
		args = parser.parse_args()

		# get path to yml file
		yml_path = args.yml
		
		# extract training and experiments from yml
		training, experiments, meta = YMLLoader.parse_yml(yml_path)

		# save args as dict
		training_args = vars(args).copy() 

		# remove any args which the user did not provide
		for arg in vars(args):
			if training_args[arg] is None:
				del training_args[arg]

		# always remove the yml arg
		if "yml" in training_args:
			del training_args["yml"]
		
		# override yml settings with command line arguments
		training.update(training_args)

		# Batch Trainer
		batchtrainer = BatchTrainer(training, experiments, meta)


if __name__ == "__main__":
	shell = Shell()	

