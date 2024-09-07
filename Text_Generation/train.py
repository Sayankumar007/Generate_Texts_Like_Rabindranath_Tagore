import os
import torch
import torch.optim as optim
import torch.nn as nn
from operator import truediv
import matplotlib.pyplot as plt
# from PIL import Image, ImageFile
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

from global_vars import *


def log_output(tr_loss, tr_acc, v_loss, v_acc, tst_loss, oa_ae, top2_acc,top5_acc, lr, epoch, EPOCHS, path):
    f = open(path, 'a')
    f.write(f"\n\n\nFor Learning_Rate : {lr} & Epochs : {EPOCHS}  (Early Stopping at {epoch})\n\nThe Result is ->\n\n")
    sentence0 = f"Train Loss : {tr_loss} | Train Acc : {tr_acc} | Valid Loss : {v_loss} | Valid Acc : {v_acc}\n"
    f.write(sentence0)
    sentence1 = 'Test Loss is: ' + str(tst_loss) + '\n' + 'OA(Top-1 Accuracy) is: ' + str(oa_ae) + '\n'
    f.write(sentence1)
    sentence3 = 'Top-2 Accuracy is: '+ str(top2_acc) + '\n'
    f.write(sentence3)
    sentence4 = 'Top-5 Accuracy is: '+ str(top5_acc) + '\n'
    f.write(sentence4)
    f.close()



def load_model(model_dict, model_path):
    if os.path.exists(model_path):
      print(f"\nLoading Saved Version of the Model {model_dict['name']} ....\n")
      model_dict['arch'].load_state_dict(torch.load(model_path)) 
    
    else :
      print(f"\nFOUND NO Previous Saved Version of the Model {model_dict['name']} ....\n")

    return model_dict['arch']  
    
  

def build_model(model_dict, dataset, train_loader, valid_loader, test_loader, EPOCHS = 500, batch_size=32, LEARNING_RATE = 0.001, measure_performance=True):
    nclass = dataset['nclass']
    epochs = EPOCHS

    # Check if the directory exists, if not, create it..
    os.makedirs(os.path.dirname(f"./{dataset['folder']}/{model_dict['folder']}/"), exist_ok=True)


    CHECKPOINT_PATH = f"./{dataset['folder']}/{model_dict['folder']}/checkpoint_{dataset['name']}_{model_dict['name']}.pth"
    model = load_model(model_dict, CHECKPOINT_PATH ).to(device)
    name = dataset['name'] + "_" + model_dict['name']

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Define the early stopping criteria
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_train_loss = float('inf')
    best_train_acc = 0.0
    patience = 10
    counter = 0

    # Lists to store accuracy and loss values
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    print(f"\nStarting Trainning... for model {model_dict['name']}\n\n")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in tqdm(train_loader, desc="Trainning : ", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc="Validating : ", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

            val_loss /= len(valid_loader.dataset)
            val_acc = val_correct / len(valid_loader.dataset)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            
        # Step the scheduler
        scheduler.step(val_loss)

        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")



        # Early stopping based on validation loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            best_val_loss = val_loss
            best_train_acc = train_acc
            best_train_loss = train_loss
            
            # Save the model if it has the best validation accuracy
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break



    # Load the best model checkpoint for evaluation
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    top2_correct=0
    top5_correct=0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing : ", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            _,top2preds = torch.topk(outputs.data,2,1)
            _,top5preds = torch.topk(outputs.data,5,1)
            test_correct += (predicted == labels).sum().item()
            for i,label in enumerate(labels):
              if label in top5preds[i]:
                top5_correct += 1
              if label in top2preds[i]:
                top2_correct += 1

        test_loss /= len(test_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)
        top2_acc = top2_correct/len(test_loader.dataset)
        top5_acc = top5_correct/len(test_loader.dataset)

    print(f"\nBEST MODEL --> \nTrain Acc : {best_train_acc:.4f} | Train Loss : {best_train_loss:.4f} | Valid Acc : {best_val_acc:.4f} | Valid Loss : {best_val_loss:.4f}")
    print(f"Test Loss : {test_loss:.4f} | Test Acc: {test_acc:.4f} | Top-2 Acc: {top2_acc:.4f} | Top-5 Acc: {top5_acc:.4f}")

    # Plot accuracy and loss curves
    plt.figure(figsize=(20, 20))
    plt.plot(train_acc_list, label='Train')
    plt.plot(val_acc_list, label='Validation')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(f"./{dataset['folder']}/{model_dict['folder']}/Acc_{name}.png")
    plt.show()


    plt.figure(figsize=(20, 20))
    plt.plot(train_loss_list, label='Train')
    plt.plot(val_loss_list, label='Validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(f"./{dataset['folder']}/{model_dict['folder']}/loss_{name}.png")
    plt.show()


    log_output(best_train_loss, best_train_acc, best_val_loss, best_val_acc, test_loss,test_acc,top2_acc,top5_acc, LEARNING_RATE, epoch, EPOCHS, f"./{dataset['folder']}/{model_dict['folder']}/results_{name}.txt")
    
    return model, [best_train_loss, best_train_acc, best_val_loss, best_val_acc, test_loss, test_acc]