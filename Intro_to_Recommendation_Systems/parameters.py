import os


#Netflix data
DATA_ROOT = '/datadrive'
NF_PRIZE_DATASET = os.path.join(DATA_ROOT, 'netflix','download','training_set') #location of extracted data
NF_DATA = 'Netflix'
TRAIN = os.path.join(NF_DATA, 'N3M_TRAIN') #os.path.join(NF_DATA, 'NF_TRAIN')
EVAL = os.path.join(NF_DATA, 'N3M_VALID') #os.path.join(NF_DATA, 'NF_VALID')
TEST = os.path.join(NF_DATA, 'N3M_TEST') #os.path.join(NF_DATA, 'NF_TEST')

#Autoencoder parameters
GPUS = 0 #'0,1,2,3'
ACTIVATION = 'selu'
OPTIMIZER = 'momentum'
HIDDEN = '512,512,1024'
BATCH_SIZE = 128
DROPOUT = 0.8
LR = 0.005
WD = 0
EPOCHS = 10
AUG_STEP = 1
MODEL_OUTPUT_DIR = 'model_save'

#Evaluation
INFER_OUTPUT = 'preds.txt'
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'model.epoch_' + str(EPOCHS-1))
MOVIE_TITLES = os.path.join('data','movie_titles.txt')




