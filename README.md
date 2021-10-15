# XAI60600
This is the project of APPLICATIONS AND PRACTICE IN NEURAL NETWORKS at the KOREA UNIV.

## 1. Project objective
The objective of the project is to verify signal processing and classification methods for motor imagery-based brain-computer interfaces. 

## 2. Brief description
Brain-computer interface (BCI) is a communication system between humans and computers. In particular, the noninvasive BCI system using the electroencephalogram (EEG) signals is one of the most notable technologies since it reflects the users’ intentions and status. Motor imagery, one of the endogenous BCI paradigms, is the paradigm in which users imagine their motions. This method can help communication of disabled by their imagination.

## 3. Dataset
BCI Competition iv 2(a)\
Download at **RELEASE**

## 4. Dataset brief info
* Number of subjects: 9 subjects
* Number of channels: 22 EEG channels(motor cortex) + 3 EOG channels
* Sampling rate: 250 Hz
* Bandpass filter: between 0.5 Hz and 100 Hz
* Class: left hand(72), right hand(72), both feet(72), and tongue(72) `total: 288 trials/sub.`
* [Detail information of dataset](http://www.bbci.de/competition/iv/desc_2a.pdf)

## 5. Evaluation
We are going to solve the subject independent task. So, the average acc. of all subjects is used as the performance indicator in the leave-one subject out(LOSO) environment. Participants use the target subject as the validation data and the source subject as the train data.
**CAUTION WE SHUFFLED THE ORIGIN TRAIN AND EVALUATION DATA**

## 6. Baseline
* Common Spatial Pattern & LinearDiscriminantAnalysis

|Sub. No.|1|2|3|4|5|6|7|8|9|avg.|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Acc.|0.455|0.267|0.319|0.326|0.205|0.340|0.281|0.524|0.389|0.345|

## 7. Project architecture
```
.
├── README.md
├── data
│   └── BCI_Competition_IV
│       └── BCI_Competition_IV_2a
│           └── A01E.gdf
│           └── A01T.gdf
│                 ...
│           └── A09E.gdf
│           └── A09T.gdf
├── .gitignore
├── baseline.ipynb
├── dataloader.py
└── sample_submission.csv
```

## 8. Envs
docker: 20.10.8

The easyest setting method\
`docker push comojin1994/cu10.2-ubuntu-18.04-pytorch-1.9.0:tagname`

If you use cuda 11.x version, contact me.
