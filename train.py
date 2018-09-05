import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from model import matchLSTM
from utils import SNLI

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/snli_dataset.pkl')

parser.add_argument('--output_dim', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--embed_dim', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=30)

parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=int, default=0.95)
parser.add_argument('--num_epochs', type=int, default=3)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print ('Loading dataloaders...')
with open(args.data_path, 'rb') as f:
    snli_dataset = pickle.load(f)

train_loader, valid_loader, test_loader = snli_dataset.get_dataloader(batch_size=args.batch_size)
print ('number of data')
print ('train', len(train_loader.dataset))
print ('valid', len(valid_loader.dataset))
print ('test', len(test_loader.dataset))

def compute_accuracy(x,y):
    _, pred = torch.max(x, dim=1)
    correct = (pred == y).float()
    accuracy = torch.mean(correct) * 100.0
    return accuracy

model = matchLSTM(args, snli_dataset.word_embedding).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

def evaluate(model, data_loader):
    model.train(False)
    model.eval()
    eval_acc = 0.
    for i, data in enumerate(data_loader):
        target = data[4].to(device)
        out = model(data[0], data[1], data[2], data[3])
        eval_acc += compute_accuracy(out, target)

    eval_acc = eval_acc.data.cpu().numpy() / (i+1)
    
    return eval_acc

def train(args, model, data_loader, criterion, optimizer):
    print ('Training...')
    best_loss = 10000.
    best_acc = 0.

    for epoch in range(args.num_epochs):
        model.train()
        train_acc = 0.
        for i, data in enumerate(data_loader):
            # forward
            target = data[4].to(device) # label
            out = model(data[0], data[1], data[2], data[3]) # prem_idx, prem_len, hypo_idx, hypo_len
            loss = criterion(out, target)

            train_acc += compute_accuracy(out, target)

            # backward and optimize
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            step_size = len(train_loader.dataset)//args.batch_size
            if (i+1) % 1000 == 0 or (i+1) == len(train_loader.dataset)//args.batch_size:
                print('Epoch [{}/{}] | Step [{}/{}] | Loss: {:.4f} | Accuracy: {:.2f}'
                        .format(epoch+1, args.num_epochs, (i+1), step_size, loss.item(), train_acc.data.cpu().numpy()/(i+1)))

        # validation
        print('Validation...')
        valid_acc = evaluate(model, valid_loader)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model.state_dict()
        
        print('Epoch [{}/{}] | Valid Accuracy: {:.2f} | Best Accuracy: {:.2f}'
                .format(epoch+1, args.num_epochs, valid_acc, best_acc))
        
        torch.save(best_model, './checkpoints/best_model.pth')
        print('Saved the best model !')

        # learning rate decay
        for params in optimizer.param_groups:
            print('set learning rate to {:.4f}'.format(params['lr']*args.lr_decay))
            params['lr'] *= args.lr_decay


train(args, model, train_loader, criterion, optimizer)
model.load_state_dict(torch.load('./checkpoints/best_model.pth'))
test_acc = evaluate(model, test_loader)
print('Test Accuracy: {:.2f}'.format(test_acc))




