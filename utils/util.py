import torch
import numpy as np

class RandCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        r_img, d_img = sample['r_img'], sample['d_img']
        score = sample['score']

        c, h, w = d_img.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        r_img = r_img[:, top: top+new_h, left: left+new_w]
        d_img = d_img[:, top: top+new_h, left: left+new_w]

        sample = {'r_img': r_img, 'd_img': d_img, 'score': score}
        return sample
    

class RandHorizontalFlip(object):
    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        r_img, d_img = sample['r_img'], sample['d_img']
        score = sample['score']

        prob_lr = np.random.random()
        # np.fliplr needs HxWxC -> transpose from CxHxW to HxWxC
        # after the flip ends, return to CxHxW
        if prob_lr > 0.5:
            r_img = np.fliplr(r_img.transpose((1, 2, 0))).copy().transpose((2, 0, 1))
            d_img = np.fliplr(d_img.transpose((1, 2, 0))).copy().transpose((2, 0, 1))

        sample = {'r_img': r_img, 'd_img': d_img, 'score': score}
        return sample


class RandRotation(object):
    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        r_img, d_img = sample['r_img'], sample['d_img']
        score = sample['score']

        prob_rot = np.random.uniform()

        if prob_rot < 0.25:     # rot0
            pass
        elif prob_rot < 0.5:    # rot90
            r_img = np.rot90(r_img.transpose((1, 2, 0))).copy().transpose((2, 0, 1))
            d_img = np.rot90(d_img.transpose((1, 2, 0))).copy().transpose((2, 0, 1))
        elif prob_rot < 0.75:   # rot180
            r_img = np.rot90(r_img.transpose((1, 2, 0)), 2).copy().transpose((2, 0, 1))
            d_img = np.rot90(d_img.transpose((1, 2, 0)), 2).copy().transpose((2, 0, 1))
        else:                   # rot270
            r_img = np.rot90(r_img.transpose((1, 2, 0)), 3).copy().transpose((2, 0, 1))   
            d_img = np.rot90(d_img.transpose((1, 2, 0)), 3).copy().transpose((2, 0, 1))
        
        sample = {'r_img': r_img, 'd_img': d_img, 'score': score}
        return sample
        


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        r_img, d_img = sample['r_img'], sample['d_img']
        score = sample['score']

        r_img = (r_img - self.mean) / self.var
        d_img = (d_img - self.mean) / self.var

        sample = {'r_img': r_img, 'd_img': d_img, 'score': score}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        # r_img: C x H x W (numpy->tensor)
        r_img, d_img = sample['r_img'], sample['d_img']
        score = sample['score']
        
        r_img = torch.from_numpy(r_img)
        d_img = torch.from_numpy(d_img)
        score = torch.from_numpy(score)

        sample = {'r_img': r_img, 'd_img': d_img, 'score': score}
        return sample


def RandShuffle(scenes, train_size=0.8):
    if scenes == "all":
        scenes = list(range(200))
    
    n_scenes = len(scenes)
    n_train_scenes = int(np.floor(n_scenes * train_size))
    n_test_scenes = n_scenes - n_train_scenes

    seed = np.random.random()
    random_seed = int(seed*10)
    np.random.seed(random_seed)
    np.random.shuffle(scenes)
    train_scene_list = scenes[:n_train_scenes]
    test_scene_list = scenes[n_train_scenes:]
    
    return train_scene_list, test_scene_list