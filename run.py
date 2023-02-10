from pyformer import Transformer
import fileio as io
import os
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

# Training data and labels
TRAIN_X, TRAIN_Y = 'data/no_fp/train/train-x.csv', 'data/no_fp/train/train-y.csv'
VAL_X, VAL_Y = 'data/no_fp/val/val-x.csv', 'data/no_fp/val/val-y.csv'
TEST_X, TEST_Y = 'data/no_fp/test/test-x.csv', 'data/no_fp/test/test-y.csv'
STD_X, STD_Y = 'data/no_fp/std/std-x.csv', 'data/no_fp/std/std-y.csv'

# Training Paramterers
EPOCH = 200
BATCH_SIZE = 32
LR = 0.0001
# Transformer Parameters
q = 8  # Query size
v = 8  # Value size
h = 4  # Number of heads
N = 2  # Number of encoder and decoder to stack
dropout = 0.2  # Dropout rate
pe = False  # Positional encoding
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_MODE"] = "offline"


def init():

    train_dataloader, shape = io.get_dataloader(TRAIN_X, TRAIN_Y, BATCH_SIZE)
    val_dataloader, _ = io.get_dataloader(VAL_X, VAL_Y, BATCH_SIZE)
    test_dataloader = io.get_test_dataloader(TEST_X, TEST_Y, BATCH_SIZE)
    std_dataloader = io.get_test_dataloader(STD_X, STD_Y, BATCH_SIZE, csv=True)

    net = Transformer(d_input=shape,
                      d_channel=1,
                      d_model=shape,
                      d_output=2,
                      q=q,
                      v=v,
                      h=h,
                      N=N,
                      dropout=dropout, pe=pe).to(DEVICE)

    # Training loop
    lossy = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)
    optimizer = optim.Adagrad(net.parameters(), lr=LR)

    for epoch in range(EPOCH):
        lossval = 0
        net.train()
        for batch_idx, (x, batch_y) in enumerate(train_dataloader):
            x, batch_y = x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = net(x.to(DEVICE))
            loss = lossy(out, batch_y.to(DEVICE))
            loss.backward()
            optimizer.step()
            lossval += loss.item()

        test_network(net, val_dataloader, lossy, "val_set")
        test_network(net, std_dataloader, lossy, "std_set")

        loss_avg = lossval/len(train_dataloader)
        print('Epoch: {} | Loss: {}\n---------------------------------------'.format(epoch,  loss_avg))

    test_network(net, test_dataloader, lossy, "test_set")
    test_network(net, std_dataloader, lossy, "std_set")



def test_network(net, dataloader_test, lossy=None, flag='test_set'):
    
    correct_list, predicted_list, actual_list = [],[],[]
    correct, total, lossval =0.0, 0.0,  0.0

    with torch.no_grad():
        for x_test, y_test in dataloader_test:
            enc_inputs, dec_inputs = x_test.to(DEVICE), y_test.to(DEVICE)
            test_outputs = net(enc_inputs)
            loss = lossy(test_outputs, dec_inputs)
            lossval += loss.item()
            _, predicted = torch.max(test_outputs.data, dim=1)

            predicted_list += predicted.tolist()
            actual_list += y_test.tolist()

            total += dec_inputs.size(0)
            correct += (predicted.float() == dec_inputs.float()).sum().item()
        if flag == 'test_set':
            correct_list.append((100 * correct / total))

    if flag == "val_set":
        print("Validation loss: ", lossval/len(dataloader_test),
              "Validation acc: ", 100 * correct // total)
    elif flag == "std_set":
        print("Stadards loss: ", lossval/len(dataloader_test),
              "Standards acc: ", 100 * correct // total)
    else:
        print("Test loss: ", lossval/len(dataloader_test),
              "Test acc: ", 100 * correct // total)

    return 100 * correct / total


if __name__ == "__main__":
    init()
