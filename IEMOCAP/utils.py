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
    
def get_video(feature_extractor, path, start_time, end_time):

    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    time_start = max(float(start_time)-1, 0)
    time_end = float(end_time)+1
    length = fps*(time_end - time_start)
    frames = []
    step = length // 8
    count = 0

    try:
        while(video.isOpened()):
            ret, image = video.read()
            if(ret==False):
                break
            if(int(video.get(1)) >= fps*time_start) and (int(video.get(1)) <= fps*time_end):
                count += 1
                if count % step == 0:
                    frames.append(image)
        video.release()
        if len(frames) >= 8:
            inputs = feature_extractor(frames[:8], return_tensors="pt")
            return inputs["pixel_values"][0]      
        else:
            lack = 8 - len(frames)
            extend_frames = [ frames[-1].copy() for _ in range(lack)]
            frames.extend(extend_frames)  
            inputs = feature_extractor(frames[:8], return_tensors="pt")
            return inputs["pixel_values"][0]

    except:
        print(f"get_video error : path - {path}, start_time - {start_time}, end_time - {end_time}, len : {len(frames)}")

def make_batchs(sessions):
    label_list = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:

        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            
            # text
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            now_speaker = speaker
            files=video_path.split("/")[-1][:-4]


        # audio
        audio_input = get_audio(audio_processor, wav_path) 
        audio_input = audio_input[-max_length:]  
        batch_audio.append(audio_input) 

        # video
        video_input = get_video(video_processor, video_path, start_time, end_time)
        batch_video.append(video_input)

        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        #innputString += " " + "</s>" + " " + prompt
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
    batch_video = padding_video(batch_video) 
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels

def teacher_batchs(sessions):
    label_list = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']

    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    for session in sessions:

        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            
            # text
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            now_speaker = speaker
            files=video_path.split("/")[-1][:-4]

        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        #innputString += " " + "</s>" + " " + prompt
        concat_string = inputString.strip()
        concat_string += " " + "</s>" + " " + prompt
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

        if len(label_list) > 3:
            label_ind = label_list.index(emotion)
        else:
            label_ind = label_list.index(sentiment)
        batch_labels.append(label_ind) 

    batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer) 
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_attention_masks, batch_labels

def audio_batchs(sessions):
    label_list = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']

    batch_input, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:

        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion = line
            
            # text
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            now_speaker = speaker
            files=video_path.split("/")[-1][:-4]

        # audio
        audio_input = get_audio(audio_processor, wav_path)
        audio_input = audio_input[-max_length:]   
        batch_audio.append(audio_input) 

        prompt = "Now"+' <s' + str(now_speaker+1) + '> '+"feels"
        #innputString += " " + "</s>" + " " + prompt
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
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_attention_masks, batch_audio, batch_labels