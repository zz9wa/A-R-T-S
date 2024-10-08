import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.classifier_getter import get_classifier
from dataset import loader
from embedding.embedding import get_embedding
from tools.tool import parse_args, print_args, set_seed

def to_tensor(data, cuda, exclude_keys=[]):
    '''
        Convert all values in the data into torch.tensor
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue

        data[key] = torch.from_numpy(data[key]).to(torch.int64)
        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data

def label2onehot(Y,args):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        I_way=nn.Parameter(torch.eye(args.way, dtype=torch.float),
                                  requires_grad=False)
        Y_onehot = F.embedding(Y.cuda(), I_way.cuda())

        return Y_onehot
def compute_w(XS, YS_onehot,args):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''
        I_support=nn.Parameter(
            torch.eye(args.shot * args.way, dtype=torch.float),
            requires_grad=False)
        lam=nn.Parameter(torch.tensor(-1, dtype=torch.float))
        W = XS.t().cuda() @ torch.inverse(
                XS.cuda() @ XS.t().cuda() + (10. ** lam.cuda()) * I_support.cuda()) @ YS_onehot.cuda()

        return W
def reidx_y(YQ,args):
        '''
            Map the labels into 0,..., way
            @param YS: batch_size
            @param YQ: batch_size

            @return YS_new: batch_size
            @return YQ_new: batch_size
        '''
        #unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

        Y_new = torch.arange(start=0, end=args.way, dtype=unique2.dtype,
                device=unique2.device)

        return Y_new[inv_Q]


def Print_Attention(file_path, vocab, model, args):
    model['G'].eval()
    word2id = vocab.itos

    data = []
    for line in open(file_path, 'r'):
        data.append(json.loads(line))

    output = {}
    output['text'] = []
    for i, temp in enumerate(data):
        output['text'].append(temp['text'])


    for i, temp in enumerate(data):
        tem = []
        length = len(temp['text'])
        for word in temp['text']:
            if word in word2id:
                tem.append(word2id.index(word))
        data[i]['text'] = np.array(tem)
        data[i]['text_len'] = 20

    data2 = {}
    data2['text'] = []
    data2['text_len'] = []
    data2['label'] = []
    for i, temp in enumerate(data):
        if temp['text'].shape[0] < 200:
            zero = torch.zeros(20 - temp['text'].shape[0])
            temp['text'] = np.concatenate((temp['text'], zero))
        else:
            temp['text'] = temp['text'][:20]
        data2['text'].append(temp['text'])
        data2['text_len'].append(temp['text_len'])
        data2['label'].append(temp['label'])

    data2['text'] = np.array(data2['text'])
    data2['text_len'] = np.array(data2['text_len'])
    data2['label'] = np.array(data2['label'])

    query = to_tensor(data2, args.cuda)
    query['is_support'] = False

    XQ, XQ_inputD, XQ_avg = model['G'](query, flag='query')
    output['attention'] = []
    for i, temp in enumerate(data):
        output['attention'].append(XQ_inputD[i].cpu().detach().numpy().tolist())
    output_file_path = 'output_attention.json'
    f_w = open(output_file_path, 'w')
    f_w.write(json.dumps(output))
    f_w.flush()
    f_w.close()

def main_attention():

    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    # initialize model
    model = {}
    model["G"], model["D"] = get_embedding(vocab, args)
    model["clf"] = get_classifier(model["G"].ebd_dim, args)

    best_path = '/home/zxs/tart_malda/bin/tmp-runs/16931211443478578/65'
    model['G'].load_state_dict(torch.load(best_path + '.G'))
    # model['D'].load_state_dict(torch.load(best_path + '.D'))
    # model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    # if args.pretrain is not None:
    #     model["ebd"] = load_model_state_dict(model["G"], args.pretrain)

    file_path = r'../data/attention_data.json'
    Print_Attention(file_path, vocab, model, args)