import torch
from torchmetrics import Accuracy
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def prediction(model:torch.nn.Module,
               test_dataloader:torch.utils.data.DataLoader,
               loss_fn,
               accuracy:Accuracy,
               device:torch.device =device):
    test_loss,test_acc= 0,0
    model.eval()
    with torch.inference_mode():
        for fig,label in test_dataloader:
            fig,label = fig.to(device),label.to(device)
            y_pred = model(fig)
            test_loss+=loss_fn(y_pred,label)
            test_acc+=accuracy(y_pred.argmax(dim=1),label)
        test_acc /=len(test_dataloader)
        test_loss /= len(test_dataloader)
    return test_acc,test_loss
