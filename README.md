# TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation (NAACL 2024)
![Figure3](https://github.com/yuntaeyang/TelME/assets/90027932/b712a639-e2cf-4cb5-a687-34ebed15afc7)
The overall flow of our model
## Requirements

Key Libraries
1. python 3.9
2. requirements.txt

## Datasets

Each data is split into train/dev/test in the [dataset folder](https://github.com/yuntaeyang/TelME/tree/main/dataset).(However, we do not provide video clip here.)
1. [MELD](https://github.com/declare-lab/MELD/)
2. [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_publication.htm)

## Train
**for MELD**
```bash
python MELD/teacher.py
python MELD/student.py
python MELD/fusion.py
```

**for IEMOCAP**
```bash
python IEMOCAP/teacher.py
python IEMOCAP/student.py
python IEMOCAP/fusion.py
```

## Testing with pretrained TelME
* Unpack model.tar.gz and place each Save_model Folder within [MELD](https://github.com/yuntaeyang/TelME/tree/main/MELD) and [IEMOCAP](https://github.com/yuntaeyang/TelME/tree/main/IEMOCAP)
```
|- MELD/
|   |- save_model/
       |- ...
```
```
|- IEMOCAP/
|   |- save_model/
       |- ...
```
- [Goole Drive](https://drive.google.com/file/d/1JIh77AqJ-mfME-nxv8r7hU3UZSrGukv0/view?usp=sharing)

## Citation

