from dataclasses import dataclass

@dataclass
class TrainingModel:

	TRAINING_AGENTS:int = 1
	ACTION_TYPE:str = "discrete"
	ACTION_SIZE:int = 7
	DEMONSTRATIONS:str = 'demonstrations'
	TRAINER:str = "trainer.txt"
	STATISTICS:str = "statistics.txt"
	PATH:str = '../unity-testing-demo-env/Assets/Resources'
	MODEL_PATH:str = 'DaggerModels'
	DEMONSTRATIONS_PATH:str = 'DaggerDemonstrations'
	TRAINER_PATH:str = 'DaggerDemonstrations'
	STATISTICS_PATH:str = 'DaggerDemonstrations'
	MODEL_NAME:str = 'kiwi'
	MAX_COMBINATIONS: int = None
	NUM_AUGS:int = None
	NUM_MODELS: int = 1
	nTraining:int = 0
	epochs:int = 300
	timestamp:str = None
	training_stats:list = None
	demonstration_ratios:list = None
	gaussian_scale:list = None


	def __post_init__(self):

		if(self.demonstration_ratios is None):
			self.demonstration_ratios = [1]

		if (self.gaussian_scale is None):
			self.gaussian_scale = [.0003]
