"""Load the data from config.ini file"""
import configparser
import os

config = configparser.ConfigParser()

# Current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Path to the config.ini file
config_path = os.path.join(dir_path, 'config.ini')
config.read(config_path)

# SCRAPER Section
streets_list = config.get('SCRAPER', 'streets_list').strip().split(',')
streets_list = [street.strip().strip("'") for street in streets_list]

# MODEL Section
model_path = config.get('MODEL', 'model_path')
label_encoder_path = config.get('MODEL', 'label_encoder_path')
backbone = config.get('MODEL', 'backbone')

# DATA Section
data_path = config.get('DATA', 'data_path')
image_size = int(config.get('DATA', 'image_size'))

# TRAINING Section
training_image_size = int(config.get('TRAINING', 'image_size'))
batch_size = int(config.get('TRAINING', 'batch_size'))
learning_rate = float(config.get('TRAINING', 'learning_rate'))
lr_min = float(config.get('TRAINING', 'lr_min'))
epochs = int(config.get('TRAINING', 'epochs'))
n_folds = int(config.get('TRAINING', 'n_folds'))
