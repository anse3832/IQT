import os
import torch
import numpy as np
import cv2 


class IQADataset(torch.utils.data.Dataset):
    def __init__(self, db_path, txt_file_name, transform, train_mode, scene_list, train_size=0.8):
        super(IQADataset, self).__init__()
        
        self.db_path = db_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.train_mode = train_mode
        self.scene_list = scene_list
        self.train_size = train_size
        
        self.data_dict = IQADatalist(
            txt_file_name=self.txt_file_name,
            train_mode=self.train_mode,
            scene_list=self.scene_list,
            train_size=self.train_size
        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])
    
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        # r_img: H x W x C -> C x H x W
        r_img_name = self.data_dict['r_img_list'][idx]
        r_img = cv2.imread(os.path.join(self.db_path, r_img_name), cv2.IMREAD_COLOR)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        r_img = np.array(r_img).astype('float32') / 255
        r_img = np.transpose(r_img, (2, 0, 1))

        # d_img: H x W x C -> C x H x W
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.db_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        
        score = self.data_dict['score_list'][idx]

        sample = {'r_img': r_img, 'd_img': d_img, 'score': score}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class IQADatalist():
    def __init__(self, txt_file_name, train_mode, scene_list, train_size=0.8):
        self.txt_file_name = txt_file_name
        self.train_mode = train_mode
        self.train_size = train_size
        self.scene_list = scene_list
        
    def load_data_dict(self):
        scn_idx_list, dist_idx_list, r_img_list, d_img_list, score_list, width_list, height_list = [], [], [], [], [], [], []

        # list append
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                scn_idx, dist_idx, ref, dis, score, width, height = line.split()                
                scn_idx = int(scn_idx)
                score = float(score)
                
                scene_list = self.scene_list

                # add items according to scene number
                if scn_idx in scene_list:
                    scn_idx_list.append(scn_idx)
                    r_img_list.append(ref)
                    d_img_list.append(dis)
                    score_list.append(score)

        # reshape score_list (1xn -> nx1)
        score_list = np.array(score_list)
        score_list = score_list.astype('float').reshape(-1, 1)

        data_dict = {'r_img_list': r_img_list, 'd_img_list': d_img_list, 'score_list': score_list}

        return data_dict
