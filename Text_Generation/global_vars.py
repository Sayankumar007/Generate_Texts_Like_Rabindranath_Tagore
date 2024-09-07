import torch
from models import BiLSTM

# general setup...
SEED = 42
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model setup
embedding_dim = 256 
hidden_dim = 256
num_layers = 2
dropout_rate = 0.5


# trainning parameters...
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
BATCH = 512
LR = 0.0001
EPOCHS = 200


# datasets...
tagore_data = dict(name="tagore", path="./data.txt", folder="Results")


# model dictionaries...
biLSTM_dict = dict(name='biLSTM', arch=BiLSTM, folder='Bi_LSTM')


# result files...
csv_list = ["loss_train.csv", "acc_train.csv", "loss_valid.csv", "acc_valid.csv", "loss_test.csv", "acc_test.csv"]



classifier_list = ['fc', 'kan']


