import os
import torch
import torchvision

from model.model_main import IQARegression
from model.backbone import inceptionresnetv2, Mixed_5b, Block35, SaveOutput
from option.config import Config

import cv2
import numpy as np

from tqdm import tqdm

# config file
config = Config({
    # device
    "GPU_ID": "0",
    
    # model for PIPAL (NTIRE2021 Challenge)
    "n_enc_seq": 21*21,                 # feature map dimension (H x W) from backbone, this size is related to crop_size
    "n_dec_seq": 21*21,                 # feature map dimension (H x W) from backbone
    "n_layer": 1,                       # number of encoder/decoder layers
    "d_hidn": 128,                      # input channel (C) of encoder / decoder (input: C x N)
    "i_pad": 0,
    "d_ff": 1024,                       # feed forward hidden layer dimension
    "d_MLP_head": 128,                  # hidden layer of final MLP 
    "n_head": 4,                        # number of head (in multi-head attention)
    "d_head": 128,                      # input channel (C) of each head (input: C x N) -> same as d_hidn
    "dropout": 0.1,                     # dropout ratio of transformer
    "emb_dropout": 0.1,                 # dropout ratio of input embedding
    "layer_norm_epsilon": 1e-12,
    "n_output": 1,                      # dimension of final prediction
    "crop_size": 192,                   # input image crop size

    # data
    "db_path": "/mnt/Dataset/anse_data/IQAdata/PIPAL/Val_Images",
    "weight_file": "epoch40.pth", # "./weights/epoch240.pth",
    "result_file": "output.txt",

    # ensemble in test
    "test_ensemble": True,
    "n_ensemble": 20
})
config.device = torch.device("cuda:%s" %config.GPU_ID if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU %s' % config.GPU_ID)
else:
    print('Using CPU')

# create_model
model_transformer = IQARegression(config).to(config.device)
model_backbone = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background').to(config.device)

# save intermediate layers
save_output = SaveOutput()
hook_handles = []
for layer in model_backbone.modules():
    if isinstance(layer, Mixed_5b):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)
    elif isinstance(layer, Block35):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

# load weights
if config.weight_file is not None:
    checkpoint = torch.load(config.weight_file)
    model_transformer.load_state_dict(checkpoint['model_state_dict'])

    model_transformer.eval()
    model_backbone.eval()
else:
    raise ValueError('You need to specify a weight file.')


# test images
filenames = os.listdir(config.db_path)
filenames.sort()
f = open(config.result_file, 'w')
for filename in tqdm(filenames):
    d_img_name = os.path.join(config.db_path, filename)
    ext = os.path.splitext(d_img_name)[-1]
    
    enc_inputs = torch.ones(1, config.n_enc_seq+1).to(config.device)
    dec_inputs = torch.ones(1, config.n_dec_seq+1).to(config.device)
    if ext == '.bmp':
        # reference image
        r_img_name = filename[:-10] + '.bmp'
        r_img = cv2.imread(os.path.join(config.db_path, 'Reference', r_img_name), cv2.IMREAD_COLOR)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        r_img = np.array(r_img).astype('float32') / 255
        r_img = (r_img - 0.5) / 0.5
        r_img = np.transpose(r_img, (2, 0, 1))
        r_img = torch.from_numpy(r_img)
        
        # distoted image
        d_img = cv2.imread(os.path.join(config.db_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = (d_img - 0.5) / 0.5
        d_img = np.transpose(d_img, (2, 0, 1))
        d_img = torch.from_numpy(d_img)

        pred = 0
        # inference (use ensemble or not)
        if config.test_ensemble:
            for i in range(config.n_ensemble):
                c, h, w = r_img.size()
                new_h = config.crop_size
                new_w = config.crop_size
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)

                r_img_crop = r_img[:, top: top+new_h, left: left+new_w].unsqueeze(0)
                d_img_crop = d_img[:, top: top+new_h, left: left+new_w].unsqueeze(0)

                r_img_crop = r_img_crop.to(config.device)
                d_img_crop = d_img_crop.to(config.device)

                # backbone feature map (ref)
                x_ref = model_backbone(r_img_crop)
                feat_ref = torch.cat(
                    (save_output.outputs[0],
                    save_output.outputs[2],
                    save_output.outputs[4],
                    save_output.outputs[6],
                    save_output.outputs[8],
                    save_output.outputs[10]),
                    dim=1
                ) # feat_ref: n_batch x (320*6) x 21 x 21
                # clear list (for saving feature map of d_img)
                save_output.outputs.clear()

                # backbone feature map (dis)
                x_dis = model_backbone(d_img_crop)
                feat_dis = torch.cat(
                    (save_output.outputs[0],
                    save_output.outputs[2],
                    save_output.outputs[4],
                    save_output.outputs[6],
                    save_output.outputs[8],
                    save_output.outputs[10]),
                    dim=1
                ) # feat_ref: n_batch x (320*6) x 21 x 21
                # clear list (for saving feature map of r_img in next iteration)
                save_output.outputs.clear()

                feat_diff = feat_ref - feat_dis
                enc_inputs_embed = feat_diff
                dec_inputs_embed = feat_ref
                pred += model_transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)
                
            pred /= config.n_ensemble
            
        else:
            c, h, w = r_img.size()
            new_h = config.crop_size
            new_w = config.crop_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            r_img_crop = r_img[:, top: top+new_h, left: left+new_w].unsqueeze(0)
            d_img_crop = d_img[:, top: top+new_h, left: left+new_w].unsqueeze(0)

            r_img_crop = r_img_crop.to(config.device)
            d_img_crop = d_img_crop.to(config.device)

            # backbone feature map (ref)
            x_ref = model_backbone(r_img_crop)
            feat_ref = torch.cat(
                (save_output.outputs[0],
                save_output.outputs[2],
                save_output.outputs[4],
                save_output.outputs[6],
                save_output.outputs[8],
                save_output.outputs[10]),
                dim=1
            ) # feat_ref: n_batch x (320*6) x 21 x 21
            # clear list (for saving feature map of d_img)
            save_output.outputs.clear()

            # backbone feature map (dis)
            x_dis = model_backbone(d_img_crop)
            feat_dis = torch.cat(
                (save_output.outputs[0],
                save_output.outputs[2],
                save_output.outputs[4],
                save_output.outputs[6],
                save_output.outputs[8],
                save_output.outputs[10]),
                dim=1
            ) # feat_ref: n_batch x (320*6) x 21 x 21
            # clear list (for saving feature map of r_img in next iteration)
            save_output.outputs.clear()

            feat_diff = feat_ref - feat_dis
            enc_inputs_embed = feat_diff
            dec_inputs_embed = feat_ref

            pred = model_transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)

        line = "%s,%f\n" % (filename, float(pred.item()))
        f.write(line)
f.close()



