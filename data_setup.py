from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
def data_setup(train_dir,test_dir,transform,test_transform,batch_size):
    train_data = ImageFolder(root=train_dir,transform=transform)
    test_data = ImageFolder(root=test_dir,transform=test_transform)

    train_datalaoder = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    test_dataloader=  DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)

    return train_datalaoder, test_dataloader
