import torch
from torch import nn
from torchmetrics import Accuracy
from tqdm.auto import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train_step(model : torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          loss_fn : torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy: Accuracy,
          device : torch.device = device):
    model.to(device)

    train_loss,train_acc = 0,0
    model.train()
    for batch_idx, (img,label) in enumerate(train_dataloader):
        img,label = img.to(device),label.to(device)
        y_pred = model(img)
        loss = loss_fn(y_pred,label)
        train_loss+= loss
        train_acc+= accuracy(y_pred.argmax(dim=1),label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    return train_loss,train_acc

def test_step(model:torch.nn.Module,
         test_dataloader:torch.utils.data.DataLoader,
         loss_fn:torch.nn.Module,
         accuracy:Accuracy,
         device : torch.device = device):
    model.to(device)

    test_loss,test_acc= 0,0
    model.eval()
    with torch.inference_mode():
        for img,label in test_dataloader:
            img,label = img.to(device),label.to(device)
            y_pred = model(img)
            test_loss+=loss_fn(y_pred,label)
            test_acc += accuracy(y_pred.argmax(dim=1),label)
        test_loss/= len(test_dataloader)
        test_acc /= len(test_dataloader)
    return test_loss,test_acc

def train(model:torch.nn.Module,
          train_dataloader:torch.utils.data,
          test_dataloader:torch.utils.data,
          loss_fn : torch.nn.Module,
          accuracy:Accuracy,
          optim : torch.optim.optimizer,
          epochs:int,
          device:torch.device = device):
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_step(model=model,
                   train_dataloader=train_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optim,
                   accuracy=accuracy,
                   device=device)
        test_loss,test_acc = test_step(model=model,
                  test_dataloader=test_dataloader,
                  loss_fn=loss_fn,
                  accuracy=accuracy,
                  device=device)
        
        print(f'Epoch: {epoch} | Train Loss: {train_loss} | Train Acc: {train_acc} \t  Test Loss: {test_loss} | Test Acc: {test_acc}')

        results['train_loss'].append(train_loss.item())
        results['train_acc'].append(train_acc.item())
        results['test_loss'].append(test_loss.item())
        results['test_acc'].append(test_acc.item())
        
    return results