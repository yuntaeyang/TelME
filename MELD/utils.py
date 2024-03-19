import torch
from transformers import RobertaTokenizer, RobertaModel, AutoProcessor, AutoImageProcessor
import librosa
import cv2
import numpy as np
import av

audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
speaker_tokens_dict = {'additional_special_tokens': speaker_list}
roberta_tokenizer.add_special_tokens(speaker_tokens_dict)

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return ids + [tokenizer.mask_token_id]

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    attention_masks = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        attention_mask = [ 1 for _ in range(len(ids))]
        add_attention = [ 0 for _ in range(len(add_ids))]
        pad_ids.append(add_ids+ids)
        attention_masks.append(add_attention+attention_mask)
    return torch.tensor(pad_ids), torch.tensor(attention_masks)

def padding_video(batch):
    max_len = 0
    for ids in batch:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in batch:
        pad_len = max_len-len(ids)
        add_ids = [ 0 for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids.tolist())
    
    return torch.tensor(pad_ids)

def get_audio(processor, path):
    audio, rate = librosa.load(path)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    return inputs["input_values"][0]
    
def get_video(feature_extractor, path):

    video = cv2.VideoCapture(path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    step = length // 8
    count = 0
    if length >= 8:

        while(video.isOpened()):
            ret, image = video.read()
            if(ret==False):
                break

            count += 1
            if count % step == 0:
                frames.append(image)
        video.release()

    else:
        while(video.isOpened()):
            ret, image = video.read()
            if(ret==False):
                break

            frames.append(image)

        video.release()
        lack = 8 - len(frames)
        extend_frames = [ frames[-1].copy() for _ in range(lack)]
        frames.extend(extend_frames)

    inputs = feature_extractor(frames[:8], return_tensors="pt")

    return inputs["pixel_values"][0]

def make_batchs(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:

        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, video_path , emotion = line
            
            # text
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            now_speaker = speaker
            files=video_path.split("/")[-1][:-4]

        audio, rate = librosa.load(video_path)
        duration = librosa.get_duration(y=audio, sr=rate)
        if duration > 30:
            batch_video.append(torch.zeros([8, 3, 224, 224]))
            batch_audio.append(torch.zeros([1412]))
        else:
            # audio
            audio_input = get_audio(audio_processor, video_path)
            audio_input = audio_input[-max_length:] 
            batch_audio.append(audio_input) 

            # video
            video_input = get_video(video_processor, video_path)
            batch_video.append(video_input)

        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))


        if len(label_list) > 3:
            label_ind = label_list.index(emotion)
        else:
            label_ind = label_list.index(sentiment)
        batch_labels.append(label_ind) 

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
    batch_audio = padding_video(batch_audio)
    batch_video = torch.stack(batch_video) 
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels
