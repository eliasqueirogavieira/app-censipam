""" This module concentrate the model creation and execution
"""

#from edet.efficientdet import model
#import edet.efficientnet
#from models.edet import backbone


from sre_constants import FAILURE
from models.app_edet import ModelEdet as efficient_det
from models.app_sfnet import ModelSFNet as sfnet
from models.app_unet import ModelUNet as unet


def load_model(config):

	try:
		Model  = eval(config.MODEL['name'])
	except:
		print("Load model: invalid model")
		exit(FAILURE)

	model_obj = Model(config)
	model_obj.load()

	return model_obj

