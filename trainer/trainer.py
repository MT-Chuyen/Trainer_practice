import torch
import numpy as np
import torchvision.transforms as T
from . import special_transforms as SegT
from model.model import UNet, my_FCN,  save_model, model_factory
from .utils import  FOCAL_LOSS_WEIGHTS, ConfusionMatrix, N_CLASSES
import torch.utils.tensorboard as tb
import time
from dataset.get_ds import get_cityscapes
import wandb
from metric.loss import loss
from metric.optimizer import otm

def get_hash(args):
    args_str = json.dumps(vars(args), sort_keys=True)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    return args_hash
 
def train(args):
    run_name = get_hash(args)
 

    if args.log:
        run = wandb.init(
            project='grasp',
            entity='truelove',
            config=args,
            name=run_name,
            force=True
        )
    
    from os import path
    model = model_factory[args.model]()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)



    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    optimizer = otm()
    w = torch.as_tensor(FOCAL_LOSS_WEIGHTS)**(-args.gamma)
    #loss = torch.nn.CrossEntropyLoss(weight=w / w.mean()).to(device) ### focal_loss
    
    loss = loss().to(device) ### without focal loss
 
    (train_ds, val_ds, test_ds, train_data, valid_data, test_data ) = get_cityscapes( )

    global_step = 0
    start = time.time()
    for epoch in range(args.num_epoch):
        print("epoch", epoch)
        model.train()
        conf = ConfusionMatrix()
        for img, label, msk in train_data:
            #print(img.shape, label.shape, msk.shape)
            img, label = img.to(device), label.to(device).long()

            logit = model(img)
            #print(img.shape) # torch.Size([32, 3, 64, 64])
            #print(label.shape) # torch.Size([32, 64, 64])
            #print(logit.shape) # torch.Size([32, 20, 24, 24])

            loss_val = loss(logit, label)
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, logit, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            conf.add(logit.argmax(1), label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if train_logger:
            train_logger.add_scalar('global_accuracy', conf.global_accuracy, global_step)
            train_logger.add_scalar('average_accuracy', conf.average_accuracy, global_step)
            train_logger.add_scalar('iou', conf.iou, global_step)
            
        
        
        model.eval()
        val_conf = ConfusionMatrix()
        for img, label, msk in valid_data:
            img, label = img.to(device), label.to(device).long()
            logit = model(img)
            val_conf.add(logit.argmax(1), label)

        if valid_logger is not None:
            log(valid_logger, img, label, logit, global_step)

        if valid_logger:
            valid_logger.add_scalar('global_accuracy', val_conf.global_accuracy, global_step)
            valid_logger.add_scalar('average_accuracy', val_conf.average_accuracy, global_step)
            valid_logger.add_scalar('iou', val_conf.iou, global_step)

        #if valid_logger is None or train_logger is None:
        print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f \t iou = %0.3f \t val iou = %0.3f' %
              (epoch, conf.global_accuracy, val_conf.global_accuracy, conf.iou, val_conf.iou))
        save_model(model)
        
    end = time.time()    
    print("Training time for this model: {:3.2f} minutes".format((end - start )/60))

 

    if args.log:
        run.log_model(path=best_model_path, name=f'{run_name}-best-model')
        run.log_model(path=last_model_path, name=f'{run_name}-last-model')

def log(logger, imgs, lbls, logits, global_step):
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(SegT.label_to_pil_image(lbls[0].cpu()).convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(SegT.label_to_pil_image(logits[0].argmax(dim=0).cpu()).convert('RGB')), global_step, dataformats='HWC')

