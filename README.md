# TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation (NAACL 2024)

![Figure3](https://github.com/yuntaeyang/TelME/assets/90027932/b712a639-e2cf-4cb5-a687-34ebed15afc7)

### Requirements

1. python 3.9
2. requirements.txt

### Datasets
Each data is split into train/dev/test in the dataset folder.(However, we do not provide video clip here.)
1. MELD
2. IEMOCAP

### Train
for MELD
python ./MELD/teacher.py
python ./MELD/student.py
python ./MELD/fusion.py

for IEMOCAP
python ./IEMOCAP/teacher.py
python ./IEMOCAP/student.py
python ./IEMOCAP/fusion.py

### Testing with pretrained TelME

* model drive
* python ./MELD/fusion.py
* python ./IEMOCAP/fusion.py
