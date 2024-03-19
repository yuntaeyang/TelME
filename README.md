# TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation (NAACL 2024)

![Figure3](https://github.com/yuntaeyang/TelME/assets/90027932/b712a639-e2cf-4cb5-a687-34ebed15afc7)

### Requirements

1. av==10.0.0
2. librosa==0.10.0
3. numpy==1.23.5
4. opencv-python==4.7.0.72
5. pandas==1.5.3
6. scikit-learn==1.2.2
7. scipy==1.10.1
8. torch==1.13.1
9. torchaudio==0.13.1
10. torchvision==0.14.1
11. transformers==4.27.2
12. python 3.9

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
