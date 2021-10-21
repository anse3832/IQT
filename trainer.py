import os

import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr


""" train model """
def train_epoch(config, epoch, model_transformer, model_backbone, save_output, criterion, optimizer, scheduler, train_loader):
    losses = []
    model_transformer.train()

    # value is not changed
    enc_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)
    dec_inputs = torch.ones(config.batch_size, config.n_dec_seq+1).to(config.device)

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        # labels: batch size 
        # enc_inputs: batch_size x len_seq+1 / dec_inputs: batch_size x len_seq+1
        # enc_inputs_embed: batch_size x len_seq x n_feats / dec_inputs_embed: batch_size x len_seq x n_feats
            
        r_img, d_img = data['r_img'].to(config.device), data['d_img'].to(config.device)
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        # backbone feature map (ref)
        x_ref = model_backbone(r_img)
        feat_ref = torch.cat(
            (save_output.outputs[0],
            save_output.outputs[2],
            save_output.outputs[4],
            save_output.outputs[6],
            save_output.outputs[8],
            save_output.outputs[10]),
            dim=1
        ) # feat_ref: n_batch x (320*6) x 29 x 29
        # clear list (for saving feature map of d_img)
        save_output.outputs.clear()

        # backbone feature map (dis)
        x_dis = model_backbone(d_img)
        feat_dis = torch.cat(
            (save_output.outputs[0],
            save_output.outputs[2],
            save_output.outputs[4],
            save_output.outputs[6],
            save_output.outputs[8],
            save_output.outputs[10]),
            dim=1
        ) # feat_ref: n_batch x (320*6) x 29 x 29
        # clear list (for saving feature map of r_img in next iteration)
        save_output.outputs.clear()

        # this value should be extracted from backbone network
        # enc_inputs_embed: batch x len_seq x n_feat
        # dec_inputs_embed: batch x len_seq x n_feat
        feat_diff = feat_ref - feat_dis
        enc_inputs_embed = feat_diff
        dec_inputs_embed = feat_ref

        # weight update
        optimizer.zero_grad()

        pred = model_transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)
        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))

    # save weights
    if (epoch+1) % config.save_freq == 0:
        weights_file_name = "epoch%d.pth" % (epoch+1)
        weights_file = os.path.join(config.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch+1))

    return np.mean(losses), rho_s, rho_p


""" validation """
def eval_epoch(config, epoch, model_transformer, model_backbone, save_output, criterion, test_loader):
    with torch.no_grad():
        losses = []
        model_transformer.eval()

        # value is not changed
        enc_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)
        dec_inputs = torch.ones(config.batch_size, config.n_dec_seq+1).to(config.device)

        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            # labels: batch size 
            # enc_inputs: batch_size x len_seq / dec_inputs: batch_size x len_seq
            # enc_inputs_embed: batch_size x len_seq x n_feats / dec_inputs_embed: batch_size x len_seq x n_feats

            if config.test_ensemble:    # use test ensemble
                r_img, d_img = data['r_img'], data['d_img']
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

                pred = 0
                for i in range(config.n_ensemble):
                    b, c, h, w = r_img.size()
                    new_h = config.crop_size
                    new_w = config.crop_size
                    top = np.random.randint(0, h - new_h)
                    left = np.random.randint(0, w - new_w)
            
                    r_img_crop = r_img[:, :, top: top+new_h, left: left+new_w].to(config.device)
                    d_img_crop = d_img[:, :, top: top+new_h, left: left+new_w].to(config.device)
             
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

                    # this value should be extracted from backbone network
                    # enc_inputs_embed: batch x len_seq x n_feat
                    # dec_inputs_embed: batch x len_seq x n_feat
                    feat_diff = feat_ref - feat_dis
                    enc_inputs_embed = feat_diff
                    dec_inputs_embed = feat_ref
                    pred = model_transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)
                
                pred /= config.n_ensemble

            else:       # do not use test ensemble                      
                r_img, d_img = data['r_img'].to(config.device), data['d_img'].to(config.device)
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

                # backbone feature map (ref)
                x_ref = model_backbone(r_img)
                feat_ref = torch.cat(
                    (save_output.outputs[0],
                    save_output.outputs[2],
                    save_output.outputs[4],
                    save_output.outputs[6],
                    save_output.outputs[8],
                    save_output.outputs[10]),
                    dim=1
                ) # feat_ref: n_batch x (320*6) x 29 x 29
                # clear list (for saving feature map of d_img)
                save_output.outputs.clear()

                # backbone feature map (dis)
                x_dis = model_backbone(d_img)
                feat_dis = torch.cat(
                    (save_output.outputs[0],
                    save_output.outputs[2],
                    save_output.outputs[4],
                    save_output.outputs[6],
                    save_output.outputs[8],
                    save_output.outputs[10]),
                    dim=1
                ) # feat_ref: n_batch x (320*6) x 29 x 29
                # clear list (for saving feature map of r_img in next iteration)
                save_output.outputs.clear()

                # this value should be extracted from backbone network
                # enc_inputs_embed: batch x len_seq x n_feat
                # dec_inputs_embed: batch x len_seq x n_feat
                feat_diff = feat_ref - feat_dis
                enc_inputs_embed = feat_diff
                dec_inputs_embed = feat_ref
                
                pred = model_transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)
            

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            loss_val = loss.item()
            losses.append(loss_val)

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        print('test epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))

        return np.mean(losses), rho_s, rho_p
