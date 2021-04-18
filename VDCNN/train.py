import os
import json
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import fire
from model.data import NaverMovieCorpus
from model.utils import JamoTokenizer
from model.network import VDCNN
from gluonnlp.data import PadSequence
from tqdm import tqdm_notebook, tqdm
from tensorboardX import SummaryWriter


def get_params(cfgpath):
    with open(cfgpath) as io:
        params = json.loads(io.read())
    return params

def load_data(params):     
    tokenizer = JamoTokenizer()
    padder = PadSequence(length=params.get('pad_length'))
    train_path = params.get('train')
    valid_path = params.get('valid')
    batch_size = params.get('batch_size')
    train_ds = NaverMovieCorpus(train_path, tokenizer, padder)
    valid_ds = NaverMovieCorpus(valid_path, tokenizer, padder)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, drop_last=False)    
    return train_dl, valid_dl


def load_model(params): 
    model = VDCNN(len(JamoTokenizer().token2idx), embedding_dim=params.get('embedding_dim'),
                  num_classes=params.get('num_classes'), k_max=params.get('k_max'))      
    return model


def evaluate(model, loss_func, valid_dl, device):
    model.eval()
    valid_loss = 0
    correct_predictions = 0
    num_predictions = 0
    for step, batch in enumerate(tqdm(valid_dl, desc = 'valid')):
        xb, yb = map(lambda x: x.to(device), batch)
        loss = loss_func(model(xb), yb)
        valid_loss += loss.item()  
        prediction = torch.max(model(xb), 1)[1]
        correct_predictions += torch.sum((yb == prediction)).item()
        num_predictions += len(yb)  
    else:
        valid_loss /= (step+1)
        valid_accuracy = correct_predictions / num_predictions
        
    return valid_loss, valid_accuracy


def save_checkpoint(model, opt, epochs, valid_accuracy, params, exp_dir):
    save_path = exp_dir + params.get('ckpt')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ckpt = {'model_state_dict': model.state_dict(),
           'epochs': epochs,
           'opt_state_dict': opt.state_dict(),
           'valid_accuracy': valid_accuracy}
    torch.save(ckpt, save_path + f'ckpt-{epochs}-{valid_accuracy:.3f}')
    print(f'saved model at {epochs}th epoch  with {valid_accuracy:.3f} valid_accuracy')
    

def main(cfgpath):
    
    params = get_params(cfgpath)
    train_dl, valid_dl  = load_data(params)
    
    model = load_model(params)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    epochs = params.get('epochs')
    learning_rate = params.get('learning_rate')
    
    
    exp_dir = os.path.split(cfgpath)[0]
    writer = SummaryWriter(log_dir = exp_dir)
    
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)    

#     print(model)

    for epoch in tqdm(range(epochs), desc='epoch'):
        
        model.train()
        train_loss  = 0
        
        for iteration, batch in enumerate(tqdm(train_dl, desc='train')):
            xb, yb = map(lambda x: x.to(device), batch)
            
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad() 
            
            train_loss  += loss
            
#             if (epoch * len(train_dl) + iteration1) % 1000 == 0:
#                 valid_loss, _ = evaluate(model, loss_func, valid_dl, device)
#                 writer.add_scalars('losses', 
#                                    {'tr_loss':train_loss/(iteration+1), 'val_loss':valid_loss},
#                                    epoch * len(train_dl) + iteration)
#                 model.train()
             
        else:
            train_loss /= (iteration + 1)

        valid_loss, valid_accuracy = evaluate(model, loss_func, valid_dl, device)
        scheduler.step(valid_loss)
        
        print(f'epoch: {epoch}, train loss: {train_loss:.3f}')
        print(f'valid loss: {valid_loss:.3f}, valid accuracy: {valid_accuracy:.3f}')

        save_checkpoint(model, opt, epoch, valid_accuracy, params, exp_dir)    
        
    writer.close()


if __name__ == '__main__':
    fire.Fire(main)

