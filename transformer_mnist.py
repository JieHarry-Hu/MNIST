import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import os.path as osp
from utils import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as Data
import random
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#from zmq import device
from vit_model import vit_base_patch16_224_in21k
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights='./weight/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
freeze_layers = True
batch_size = 256
lr=0.01
lrf=0.01
epochs=5
log_save_root_path = "./log/transformer/"
model_save_root_path = "./model/transformer/"
data_tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize([224, 224])]
)
train_dataset = datasets.MNIST(root='./',train=True,transform=data_tf,download=True)
test_dataset = datasets.MNIST(root='./',train=False,transform=data_tf,download=True)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True
)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True
)
model = vit_base_patch16_224_in21k(num_classes=10, has_logits=False).to(device)

if weights != "":
    assert os.path.exists(weights), "weights file: '{}' not exist.".format(weights)
    weights_dict = torch.load(weights, map_location=device)
    # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=False))
if freeze_layers:
    for name, para in model.named_parameters():
        # 除head, pre_logits外，其他权重全部冻结
        if "head" not in name and "pre_logits" not in name:
            para.requires_grad_(False)
        else:
            print("training {}".format(name))
pg = [p for p in model.parameters() if p.requires_grad]
#optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
optimizer = optim.AdamW(pg, lr=lr, weight_decay=0.01)
lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
loss_fun = nn.CrossEntropyLoss()
def train(epochs,model,train_loader,optimizer,loss_fn):
    global epoch_acc
    writer = SummaryWriter('runs/trans_mnist_experiment')
    log = open(osp.join(log_save_root_path, 'trans_{}_{}.txt'.format('AdamW', 'crossentropyloss')),
               'w')
    print_log('training...',log)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        t1=time.time()
        epoch_acc=0.
        epoch_loss=0.
        acc_log=[]
        loss_log=[]
        print_log('epoch : [{}/{}]'.format(epoch+1, epochs), log)
        loop = tqdm(enumerate(train_loader),total=len(train_loader))
        for i,(data, target) in loop:
            data,target = data.to(device),target.to(device)
            out = model(data)
            loss = loss_fn(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            output = torch.max(out, dim=1)[1]
            right = torch.sum(output==target).item()
            epoch_acc+=right
            log.write('epoch:{} loss:{:.4f} acc:{:.4f}%\n'.format(epoch+1,loss.item(),100*right/batch_size))
            log.flush()
            acc_log.append(100*right/batch_size)
            loss_log.append(loss.item())
            loop.set_description('epoch:{} loss:{:.4f} acc:{:.4f}%'.format(epoch+1,loss.item(),100*right/batch_size))
            writer.add_scalar('training loss',loss.item(),epoch*len(train_loader)+i)
            writer.add_scalar('training acc',100*right/batch_size,epoch*len(train_loader)+i)
        loop.close()
        loop.close()
        epoch_acc/=len(train_loader.dataset)
        epoch_loss/=len(train_loader.dataset)
        print_log('epoch_acc : {:.4f}%'.format(epoch_acc), log)
        print_log('epoch_loss : {:.4f}'.format(epoch_loss), log)
        print_log('saving model...', log)
        torch.save(model.state_dict(), osp.join(model_save_root_path, 'transformer_{}_{}_acc:{:.4f}.pth'.format('AdamW',epoch+1, epoch_acc)))
        print_log('saving model done', log)
        t2=time.time()
        print_log('time : {:.4f}'.format(t2-t1), log)
    log.close()
    writer.close()
    print('training done')
    print('epoch_acc:', epoch_acc)
    print('epoch_loss:', epoch_loss)
def test(model,test_loader,weight_path):
    log = open(osp.join(log_save_root_path, 'test_trans_AdamW.txt'),
               'w')
    print_log('testing...',log)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    right = 0
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader),total=len(test_loader))
        for i,(x,y) in loop:
            x,y = x.to(device),y.to(device)
            out = model(x)
            output = torch.max(out, dim=1)[1]
            right += torch.sum(output==y).item()
            log.write('acc:{:.4f}%\n'.format(100*right/batch_size))
            log.flush()
            loop.set_description('acc:{:.4f}%'.format(100*right/len(test_loader.dataset)))
        loop.close()
        print_log('acc : {:.4f}%'.format(right/len(test_loader.dataset)), log)
    log.close()
    print('testing done')
if __name__ == '__main__':
    train(epochs,model,train_loader,optimizer,loss_fun)
    test_model = model
    test(test_model,test_loader,osp.join(model_save_root_path, 'transformer_{}_{}_acc:{:.4f}.pth'.format('AdamW',epochs, epoch_acc)))
