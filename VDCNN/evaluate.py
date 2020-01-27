import os
import glob
import torch
from model.data import NaverMovieCorpus
from model.utils import JamoTokenizer
from model.network import VDCNN
import json
from gluonnlp.data import PadSequence
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook, tqdm
import fire


def get_params(cfgpath):
    with open(cfgpath) as io:
        params = json.loads(io.read())
    return params

def load_data(params):     
    tokenizer = JamoTokenizer()
    padder = PadSequence(length=params.get('pad_length'))
    batch_size = params.get('batch_size')
    test_path = params.get('test')
    test_ds = NaverMovieCorpus(test_path, tokenizer, padder)
    test_dl = DataLoader(test_ds, batch_size*2, drop_last=False)    
    return test_dl

def load_model(cfgpath, params): 
    model = VDCNN(len(JamoTokenizer().token2idx), 
                  embedding_dim=params.get('embedding_dim'),
                  num_classes=params.get('num_classes'), 
                  k_max=params.get('k_max')) 
    
    exp_dir = os.path.split(cfgpath)[0]
    model_path = exp_dir + params.get('ckpt')
    latest_model = max(glob.glob(model_path + '*'), key=os.path.getctime)
    print(f'model: {latest_model}')
    ckpt = torch.load(latest_model)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval() 
    return model

def eval(cfgpath):
    params = get_params(cfgpath)
    test_dl = load_data(params)

    model = load_model(cfgpath, params)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    correct = 0
       
    for iteration, batch in enumerate(tqdm(test_dl, desc='test')):
        xb, yb = map(lambda x: x.to(device), batch)
        output = model(xb)
        _, prediction = torch.max(output, 1)
        correct += torch.sum((yb == prediction)).item()

    accuracy = correct / len(test_dl.dataset)
    print("test accuracy : {:.3f}".format(accuracy))

if __name__ == '__main__':
    fire.Fire(eval)
