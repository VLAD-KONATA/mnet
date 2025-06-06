import config
args, unparsed = config.get_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
from tqdm import tqdm

from data import trainSet
from util_evaluation import calc_psnr,calc_ssim
import datetime
from select_model import select_model
import optim
#import val
from select_loss import Select_Loss
import torch.nn.functional as F
from losses.laplacianpyramid import *

from opt import *
import torch.nn as nn
import json
from importlib import import_module

def args_add_additinoal_attr(args,json_path):
    dic = json.load(open(json_path,'r',))
    for key,value in dic.items():
        if key == '//':
            continue
        setattr(args,key,value)

def select_tmodel(args,student,conv=None):
    opt_path = f'opt/{args.model}.json'
    args_add_additinoal_attr(args, opt_path)
    module = import_module(f'model_zoo.{args.model.lower()}.basic_model')
    model = module.make_model(args,student,conv)
    return model
####################################################################
# seed
GLOBAL_SEED = 777
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
student=True

args.ckpt_dir = 'experiments/'+args.model+'/'+args.ckpt_dir
os.makedirs(args.ckpt_dir,exist_ok=True)

if len(args.gpu_id) > 1:
    args.parallel = True

# data
trainset = trainSet(data_root=args.traindata_path,args=args)
# batch_size = args.batch_size*len(device_ids)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                shuffle=False,num_workers=args.num_workers, pin_memory=False)
if student:
    smodel = select_model(args,student)
    # model
    tmodel = select_tmodel(args,student=False)
    checkpoint = torch.load('/home/konata/Git/mnet/experiments/mamba_unet_dist/IXI_x2_2x4b/pth/0500.pth', map_location=torch.device('cpu'))
    tmodel.load_state_dict(checkpoint['state_dict'])
    smodel = smodel.cuda()
else:
    tmodel=select_model(args,student)




tmodel = tmodel.cuda()

#### optim ####
if student:
    optimizer = torch.optim.Adam(smodel.parameters(), lr=args.lr)
else:
    optimizer = optim.select_optim(args,tmodel)
# optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
scheduler = optim.select_scheduler(args,optimizer)


    # 定义优化器
#### loss ####
hard_loss = nn.L1Loss()
soft_loss = nn.KLDivLoss(reduction="batchmean")
l = nn.SmoothL1Loss(reduction="mean")
#soft_loss = LaplacianLoss()
loss_function = Select_Loss(args).cuda()

########################### train ###################################
# log
with open(args.ckpt_dir + '/logs.txt',mode='a+') as f:
    s = "START EXPERIMENT\n"
    f.write(s)
    for i,j in args.__dict__.items():
        f.write(str(i)+' : '+str(j)+'\n')

# amp
if args.amp:
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

temp=7
alpha=0.3

if student:
    smodel.train()
else:
    tmodel.train()
for epoch in tqdm(range(args.start_epoch,args.max_epoch)):
    loss_epoch = 0
    psnr_epoch = 0

    for iter, hr in tqdm(enumerate(dataloader)):
                 
        if len(hr.shape) == 5:
            gt = torch.cat([i for i in hr],0) #[bz,h,w,7]
        else: gt = hr
        lr = gt[...,::args.upscale] # [bz,h,w,4]
        if torch.cuda.is_available():
            gt = Variable(gt.cuda())
            lr = Variable(lr.cuda())
        
        optimizer.zero_grad()
        if student:
            if args.amp:
                with autocast():
                    with torch.no_grad():
                        tsr,tmiddle = tmodel(lr,False)
                    #print('pred_tmodel')
                    # 学生模型预测
                    ssr,smiddle = smodel(lr,True)
                    #print('pred_smodel')

                    # 计算hard_loss
                    student_hard_loss = hard_loss(ssr,gt)
                    my_loss = l(smiddle, tmiddle)
                    # chatgpt版Loss
                    soft_student_outputs = F.log_softmax(ssr / temp, dim=1)
                    soft_teacher_outputs = F.softmax(tsr/temp,dim=1)

                    ditillation_loss = soft_loss(soft_student_outputs,soft_teacher_outputs)
                    loss = alpha * student_hard_loss + (1-alpha) * temp * temp * ditillation_loss+my_loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                sr = smodel(lr)
                '''
                loss_iter = loss_function(sr,gt)
                loss_iter.backward()
                '''
                optimizer.step()
        else:
            if args.amp:
                with autocast():
                    tsr,tmiddle = tmodel(lr,student)
                    loss = loss_function(tsr,gt)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                sr = smodel(lr)
                '''
                loss_iter = loss_function(sr,gt)
                loss_iter.backward()
                '''
                optimizer.step()

        psnr_iter = 0
        for bz in range(gt.shape[0]):
            if student:
                psnr_iter += calc_psnr(gt[bz, :, :, :],ssr[bz, :, :, :]).item()
            else:
                psnr_iter += calc_psnr(gt[bz, :, :, :],tsr[bz, :, :, :]).item()

        psnr_iter /= (bz+1)  

        #### log ####    
        lr_tmp = optimizer.state_dict()['param_groups'][0]['lr']
        log = r"epoch[{}/{}] iter[{}/{}] psnrTr:{:.6f} lossTr:{:.12f} lr:{:.12f}"\
            .format(epoch+1, args.max_epoch , \
                iter+1, len(dataloader),\
                psnr_iter,loss,lr_tmp)
        now = str(datetime.datetime.now())
        print(now+' '+log)

        loss_epoch += loss
        psnr_epoch += psnr_iter

    loss_epoch /= (iter+1)
    psnr_epoch /= (iter+1)

    #### lr schedule ####
    if args.schedule == 'step':
        scheduler.step()
    elif args.schedule == 'cos_lr':
        #### torch.cos_lr ####
        # scheduler.step()
        #### timm.cos_lr ####
        scheduler.step_update(epoch)
    elif args.schedule == 'Tmin':
        scheduler.step(loss_epoch)
    elif args.schedule == 'Tmax':
        scheduler.step(psnr_epoch)

    epoch +=1


    log = r"epoch[{}/{}] psnrTr:{:.6f} lossTr:{:.12f} lr:{:.12f}"\
    .format(epoch, args.max_epoch , \
        psnr_epoch,loss_epoch,lr_tmp)
    now = str(datetime.datetime.now())
    print(now+' '+log)
    with open(args.ckpt_dir + '/logs.txt',mode='a+') as f:
        f.write('\n'+now+log)

    if epoch >int(0.99*args.max_epoch):
        os.makedirs(args.ckpt_dir+'/pth',exist_ok=True)
        os.makedirs(args.ckpt_dir+'/keys',exist_ok=True)
        if student:
            model=smodel
        else:
            model=tmodel
        try:
            torch.save({'epoch': epoch, 'state_dict': model.module.state_dict()}, args.ckpt_dir + '/pth/' + str(epoch).zfill(4) + '.pth')
        except:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, args.ckpt_dir + '/pth/' + str(epoch).zfill(4) + '.pth')


# val_opt = True
# if val_opt:
#     val.val(args=args,model=model)

