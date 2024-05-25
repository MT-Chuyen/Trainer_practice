from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader

root = '/media/mountHDD2/chuyenmt/cityscapes'
def get_cityscapes(root):
 
    train_ds= Cityscapes(root, split='train', mode='fine',target_type='semantic')
    val_ds = Cityscapes(root, split='val', mode='fine',target_type='semantic')
    test_ds = Cityscapes(root, split='test', mode='fine',target_type='semantic')
    
    train_dl = DataLoader(train_ds, batch_size=32 )
    valid_dl = DataLoader(val_ds, batch_size=32 )
    test_dl = DataLoader(test_ds, batch_size=32 )
 

    return (train_ds, val_ds, test_ds, train_dl, valid_dl, test_dl ) 