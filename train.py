import os
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader

from model.model_main import IQARegression
from model.backbone import inceptionresnetv2, Mixed_5b, Block35, SaveOutput
from option.config import Config
from trainer import train_epoch, eval_epoch
from utils.util import RandCrop, RandHorizontalFlip, RandRotation, Normalize, ToTensor, RandShuffle

# config file
config = Config({
    # device
    "GPU_ID": "0",
    "num_workers": 8,

    # model for LIVE/CSIQ/TID
    # "n_enc_seq": 29*29,               # feature map dimension (H x W) from backbone, this size is related to crop_size
    # "n_dec_seq": 29*29,               # feature map dimension (H x W) from backbone
    # "n_layer": 2,                     # number of encoder/decoder layers
    # "d_hidn": 256,                    # input channel (C) of encoder / decoder (input: C x N)
    # "i_pad": 0,
    # "d_ff": 1024,                     # feed forward hidden layer dimension
    # "d_MLP_head": 512,                # hidden layer of final MLP 
    # "n_head": 4,                      # number of head (in multi-head attention)
    # "d_head": 256,                    # input channel (C) of each head (input: C x N) -> same as d_hidn
    #"dropout": 0.1,                   # dropout ratio of transformer
    # "emb_dropout": 0.1,               # dropout ratio of input embedding
    # "layer_norm_epsilon": 1e-12,
    # "n_output": 1,                    # dimension of final prediction
    # "crop_size": 256,                 # input image crop size                 

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
    "db_name": "PIPAL",                                                     # database name [ PIPAL | LIVE | CSIQ | TID2013 ]
    "db_path": "/mnt/Dataset/anse_data/IQAdata/PIPAL/Train_Images",         # root of dataset
    "snap_path": "./weights/PIPAL",                                         # path for saving weights
    "txt_file_name": "./IQA_list/PIPAL_augment.txt",                    # image list file (.txt)
    "train_size": 0.8,
    "scenes": "all",
    
    # ensemble in validation phase
    "test_ensemble": True,
    "n_ensemble": 5,

    # optimization
    "batch_size": 16,
    "learning_rate": 2e-4,
    "weight_decay": 1e-5,
    "n_epoch": 300,
    "val_freq": 1,
    "save_freq": 5,
    "checkpoint": None,                 # load pretrained weights
    "T_max": 50,                        # cosine learning rate period (iteration)
    "eta_min": 0                        # mininum learning rate
})

# device setting
config.device = torch.device("cuda:%s" %config.GPU_ID if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU %s' % config.GPU_ID)
else:
    print('Using CPU')

# data selection
if config.db_name == 'PIPAL':
    from data.data_PIPAL import IQADataset
elif config.db_name == 'LIVE':
    from data.data_LIVE import IQADataset

# data separation (8:2)
train_scene_list, test_scene_list = RandShuffle(config.scenes, config.train_size)
# for reproducing results
# train_scene_list = [*range(160)]
# test_scene_list = [*range(160, 200)]
# train_scene_list = [*range(23)]
# test_scene_list = [*range(23, 29)]
print('number of train scenes: %d' % len(train_scene_list))
print('number of test scenes: %d' % len(test_scene_list))

# data load
train_dataset = IQADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    transform=transforms.Compose([RandCrop(config.crop_size), Normalize(0.5, 0.5), RandHorizontalFlip(), RandRotation(), ToTensor()]),
    train_mode=True,
    scene_list=train_scene_list,
    train_size=config.train_size
)
test_dataset = IQADataset(
    db_path=config.db_path,
    txt_file_name=config.txt_file_name,
    transform= transforms.Compose([Normalize(0.5, 0.5), ToTensor()]) if config.test_ensemble else transforms.Compose([RandCrop(config.crop_size), Normalize(0.5, 0.5), ToTensor()]),
    train_mode=False,
    scene_list=test_scene_list,
    train_size=config.train_size
)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)


# create model
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

# loss function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model_transformer.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0

# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)

# train & validation
losses, scores = [], []
for epoch in range(start_epoch, config.n_epoch):
    loss, rho_s, rho_p = train_epoch(config, epoch, model_transformer, model_backbone, save_output, criterion, optimizer, scheduler, train_loader)
    # print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))

    if (epoch+1) % config.val_freq == 0:
        loss, rho_s, rho_p = eval_epoch(config, epoch, model_transformer, model_backbone, save_output, criterion, test_loader)
        # print('test epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))
    