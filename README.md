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
- [Naver drive]

## Citation

