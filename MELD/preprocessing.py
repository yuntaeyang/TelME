import csv

def split(session):
    final_data = []
    split_session = []
    for line in session:
        split_session.append(line)
        final_data.append(split_session[:])    
    return final_data

def preprocessing(data_path):
  f = open(data_path, 'r')
  rdr = csv.reader(f)
        
  session_dataset = []
  session = []
  speaker_set = []

  pre_sess = 'start'
  for i, line in enumerate(rdr):
    if i == 0:
      header  = line
      utt_idx = header.index('Utterance')
      speaker_idx = header.index('Speaker')
      emo_idx = header.index('Emotion')
      sess_idx = header.index('Dialogue_ID')
      video_idx = header.index('Video_Path')

    else:
      utt = line[utt_idx]
      speaker = line[speaker_idx]
      if speaker in speaker_set:
        uniq_speaker = speaker_set.index(speaker)
      else:
        speaker_set.append(speaker)
        uniq_speaker = speaker_set.index(speaker)
      emotion = line[emo_idx]
      sess = line[sess_idx]
      video_path = line[video_idx]

      if pre_sess == 'start' or sess == pre_sess:
        session.append([uniq_speaker, utt, video_path, emotion])

      else:
        session_dataset += split(session)
        session = [[uniq_speaker, utt, video_path, emotion]]
        speaker_set = []
      pre_sess = sess   
  """ 마지막 세션 저장 """
  session_dataset += split(session)           
  f.close()
  return session_dataset