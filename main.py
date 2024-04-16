
from dataset import MNIST
from model import LeNet5, CustomMLP, LeNet5_regularized

# import some packages you need here
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, trn_loader, tst_loader, device, criterion, optimizer, epochs):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        acc = 0.0
        total_loss = 0.0
        correct = 0.0
        total_samples = 0
        for img, label in tqdm(trn_loader):
            optimizer.zero_grad()
            total_samples += label.size(0)
            
            img = img.to(device)
            label = label.to(device)
                        
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predict = torch.argmax(out, dim = -1)
            correct += torch.sum(predict == label).item()   
            
        acc = correct / total_samples
        train_acc.append(acc)
        
        total_loss = total_loss / len(trn_loader)
        train_loss.append(total_loss)
        
        eval_loss, eval_acc = test(model, tst_loader, device, criterion)
        valid_acc.append(eval_acc)
        valid_loss.append(eval_loss)
        
        print("{} epoch loss:{}, accuracy:{}".format(epoch+1, round(total_loss, 3), round(acc, 3)))
        print("{} epoch eval_loss:{}, eval_accuracy:{}".format(epoch+1, eval_loss, eval_acc))
        print("-"*50)

    return train_loss, train_acc, valid_loss, valid_acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model = model.to(device)
    model.eval()
    acc = 0.0
    total_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for img, label in tqdm(tst_loader):

            total_samples += label.size(0)
            
            img = img.to(device)
            label = label.to(device)
                        
            out = model(img)
            loss = criterion(out, label)
            
            total_loss += loss.item()
            predict = torch.argmax(out, dim = -1)
            correct += torch.sum(predict == label).item()
               
        acc = correct / total_samples
        total_loss = total_loss / len(tst_loader)
        acc = round(acc, 3)
        total_loss = round(total_loss, 3) 
        

    return total_loss, acc

def loader(dir_path, batch, type = None, model = None):
    dataset = MNIST(dir_path, model = model)
    if type == "train":
        loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers = 4)
    if type == "test":
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers = 4)
    return loader

def plot(data, label, model, epochs):
    epochs = [int(i+1) for i in range(epochs)]
    plt.plot(epochs, data, marker = 'o')
    plt.xlabel("Epoch")
    plt.ylabel(model + label)
    plt.title(model + label)
    plt.savefig("C:/Users/noble/Downloads/mnist-classification/mnist-classification/plot"+ model +"_"+ label+".png")
    plt.clf()

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    device = "cpu"
    epochs = 20
    batch_size = 64
    train_dir = "C:/Users/noble/Downloads/mnist-classification/mnist-classification/data/test/test"
    test_dir =  "C:/Users/noble/Downloads/mnist-classification/mnist-classification/data/train/train"
    
    LeNet5_train_loader = loader(train_dir, batch = batch_size, type = "train", model = "LeNet5")
    LeNet5_test_loader = loader(test_dir, batch = batch_size, type = "test", model = "LeNet5")
    
    MLP_train_loader = loader(train_dir, batch = batch_size, type = "train", model = "CustomMLP")
    MLP_test_loader = loader(test_dir, batch = batch_size, type = "test", model = "CustomMLP")
    
    
    LeNet5_model = LeNet5()
    LeNet5_optimizer = torch.optim.SGD(LeNet5_model.parameters(), lr=0.01, momentum = 0.9)
    print("LeNet5 :", sum(p.numel() for p in LeNet5_model.parameters() if p.requires_grad))
    
    LeNet5R_model = LeNet5_regularized()
    LeNet5R_optimizer = torch.optim.SGD(LeNet5R_model.parameters(), lr=0.01, momentum = 0.9)
    print("LeNet5R :", sum(p.numel() for p in LeNet5R_model.parameters() if p.requires_grad))
    
    MLP_input_size = len(MLP_train_loader.dataset[0][0][-1])
    MLP_model = CustomMLP(MLP_input_size)
    MLP_optimizer = torch.optim.SGD(MLP_model.parameters(), lr=0.01, momentum = 0.9)
    print("MLP_model :", sum(p.numel() for p in MLP_model.parameters() if p.requires_grad))

    
    
    criterion = nn.CrossEntropyLoss()
    
    LeNet5_train_loss, LeNet5_train_acc, LeNet5_valid_loss, LeNet5_valid_acc = train(LeNet5_model, LeNet5_train_loader, LeNet5_test_loader, device, criterion, LeNet5_optimizer, epochs)
    plot(LeNet5_train_loss, "loss", "LeNet5_train", epochs)
    plot(LeNet5_train_acc, "acc", "LeNet5_train", epochs)
    plot(LeNet5_valid_loss, "loss", "LeNet5_test", epochs)
    plot(LeNet5_valid_acc, "acc", "LeNet5_test", epochs)
    
    LeNet5R_train_loss, LeNet5R_train_acc, LeNet5R_valid_loss, LeNet5R_valid_acc = train(LeNet5R_model, LeNet5_train_loader, LeNet5_test_loader, device, criterion, LeNet5R_optimizer, epochs)
    plot(LeNet5R_train_loss, "loss", "LeNet5R_train", epochs)
    plot(LeNet5R_train_acc, "acc", "LeNet5R_train", epochs)
    plot(LeNet5R_valid_loss, "loss", "LeNet5R_test", epochs)
    plot(LeNet5R_valid_acc, "acc", "LeNet5R_test", epochs)
    
    MLP_train_loss, MLP_train_acc, MLP_valid_loss, MLP_valid_acc = train(MLP_model, MLP_train_loader, MLP_test_loader, device, criterion, MLP_optimizer, epochs)
    plot(MLP_train_loss, "loss", "MLP_train", epochs)
    plot(MLP_train_acc, "acc", "MLP_train", epochs)
    plot(MLP_valid_loss, "loss", "MLP_test", epochs)
    plot(MLP_valid_acc, "acc", "MLP_test", epochs)

if __name__ == '__main__':
    main()
