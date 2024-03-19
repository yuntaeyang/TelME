import glob
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import gc

from preprocessing import *
from utils import *
from dataset import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def CELoss(pred_outs, labels):
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def model_train(training_epochs, model_t, audio_s, video_s, fusion, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    best_dev_fscore, best_test_fscore = 0, 0   
    best_epoch = 0

    for epoch in tqdm(range(training_epochs)):
        fusion.train() 
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                """Prediction"""
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
                text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
                audio_hidden, audio_logits = audio_s(audio_inputs)
                video_hidden, video_logits = video_s(video_inputs)
                
                pred_logits = fusion(text_hidden, audio_hidden, video_hidden)

                loss_val = CELoss(pred_logits, batch_labels)
            
            scaler.scale(loss_val).backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        fusion.eval()   
        dev_pred_list, dev_label_list = evaluation(model_t, audio_s, video_s, fusion, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        print(f"dev_score : {dev_fbeta}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_epoch = epoch
            _SaveModel(fusion, save_path)

            fusion.eval()
            test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, fusion, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                
            print(f"test_score : {test_fbeta}")

def evaluation(model_t, audio_s, video_s, fusion, dataloader):
    fusion.eval()
    label_list = []
    pred_list = []
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden, video_logits = video_s(video_inputs)
                
            pred_logits = fusion(text_hidden, audio_hidden, video_hidden)
            
            """Calculation"""    

            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'total_fusion.bin'))

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3
    """Dataset Loading"""
    
    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"

    data_path = './dataset/MELD.Raw/'

    train_path = data_path + 'train_meld_emo.csv'
    dev_path = data_path + 'dev_meld_emo.csv'
    test_path = data_path + 'test_meld_emo.csv'


    train_dataset = meld_dataset(preprocessing(train_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=make_batchs)


    dev_dataset = meld_dataset(preprocessing(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    test_dataset = meld_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    save_path = os.path.join('./MELD/save_model')
    
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    clsNum = len(train_dataset.emoList)
    init_config = Config()

    '''teacher model load'''
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load('./MELD/save_model/teacher.bin'))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()

    '''student model'''
    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load('./MELD/save_model/student_audio/total_student.bin')) 
    for para in audio_s.parameters():
        para.requires_grad = False
    audio_s = audio_s.cuda()
    audio_s.eval()

    video_s = Student_Video(video_model, clsNum)
    video_s.load_state_dict(torch.load('./MELD/save_model/student_video/total_student.bin')) 
    for para in video_s.parameters():
        para.requires_grad = False
    video_s = video_s.cuda()
    video_s.eval()

    '''fusion'''
    hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion = fusion.cuda()
    fusion.eval()


    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    model_train(training_epochs, model_t, audio_s, video_s, fusion, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, scaler, save_path)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)