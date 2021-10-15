from glob import glob
from tqdm import tqdm
import mne
import torch
import scipy
import numpy as np


class BCICompet2aIV(torch.utils.data.Dataset):
    def __init__(self,
                 base_path: str,
                 target_subject: int,
                 is_subject_independent: bool=False,
                 is_val: bool=False,
                is_test: bool=False):
        
        '''
        * 769: Left
        * 770: Right
        * 771: foot
        * 772: tongue
        '''
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = base_path
        self.target_subject = target_subject
        self.is_subject_independent = is_subject_independent
        self.is_val = is_val
        self.is_test = is_test
        
        self.data, self.label = self.get_brain_data()
        
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        data = self.data[index, ...]
        label = self.label[index]
        
        sample = {'data': data, 'label': label}
        
        return sample
    
    
    def get_brain_data(self):
        filelist = sorted(glob(f'{self.base_path}/*T*.gdf')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E*.gdf'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            
            if self.is_subject_independent and not (self.is_val or self.is_test):
                # subject independent이고 train set
                if idx == self.target_subject: continue
            elif self.is_subject_independent and (self.is_val or self.is_test):
                # subject independent이고 test set
                if idx != self.target_subject: continue
            elif not self.is_subject_independent:
                # subject dependent일 때
                if idx != self.target_subject: continue
                    
            print(f'LOG >>> Filename: {filename}')
            
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, annot = mne.events_from_annotations(raw)
            
            raw.load_data()
            raw.filter(7., 35., fir_design='firwin')
            raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
            
            picks = mne.pick_types(raw.info,
                                   meg=False,
                                   eeg=True,
                                   eog=False,
                                   stim=False,
                                   exclude='bads')
            
            tmin, tmax = 0., 3.
            if not self.is_test:
                event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10}) if idx != 3 \
                else dict({'769': 5,'770': 6,'771': 7,'772': 8})
            else:
                event_id = dict({'783': 7})
            
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin,
                                tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                preload=True)
            
            splited_data = epochs.get_data()
            splited_data = splited_data[:, np.newaxis, ...]
            label_list = epochs.events[:,-1] - 7
            
            if self.is_test:
                label_list = np.zeros(len(data))
            
            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)
                
        return data, label