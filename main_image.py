import torch
from net.model_LADMM import LADMM
import time
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.ImageDataset import ImageFileDataset
import argparse
import logging
from torch.optim.lr_scheduler import MultiStepLR
import datetime
from meter import AverageMeter, Summary, ProgressMeter
import numpy as np
import lpips


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.switch_backend('agg')

from torch.utils.tensorboard import SummaryWriter
 
def train(dataloader, model, loss_fn, optimizer, epoch, device, args, writer:SummaryWriter):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses], prefix="Epoch: [{}]".format(epoch))
    model.train()

    end = time.time()
    for batch, (x, y) in enumerate(dataloader):
        data_time.update(time.time() - end)
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss1 = loss_fn[0](pred, y) + + 1e-6
        # with torch.no_grad():
        loss2 = torch.mean(loss_fn[1](pred, y) + 1e-6)
        gamma = 10
        if epoch > 20:
            gamma = 1
        loss = loss1 * gamma + loss2 

        logging.debug("loss per batch:{} = {} * {} + {}\n".format(loss, loss1, gamma, loss2))
        losses.update(loss.item(), x.size(0))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % 100 == 0:
            progress.display(batch)
        
        writer.add_scalar('train/loss', loss, epoch)  # 画loss，横坐标为epoch
        writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch)

    return losses.avg
    
def val(dataloader, model, loss_fn, epoch, device, writer):
    losses = AverageMeter('Loss', ':.6f', Summary.NONE)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += (torch.mean(loss_fn[1](pred, y))).item() 

    test_loss /= num_batches
    losses.update(test_loss, 1)
    logging.info(f"Val: Avg loss: {test_loss:>8f}")
    writer.add_scalar('val/loss', test_loss, epoch)  # 画loss，横坐标为epoch
    return test_loss


def test(dataloader, model, device, epoch, args):
    model.eval()
    num = 0  
    with torch.no_grad():
        for x, y in dataloader:
            num = num + 1
            x, y = x.to(device), y.to(device)
            pred = model(x)
            if num > 3:
                break      
            img = x.detach().squeeze(0).permute(1,2,0).cpu().numpy()
            rec = pred.detach().squeeze(0).permute(1,2,0).cpu().numpy()
            target = y.detach().squeeze(0).permute(1,2,0).cpu().numpy()

            img /= np.max(img)
            rec /= np.max(rec)
            target /= np.max(target)

            plt.rcParams['figure.figsize'] = (20, 4.0)
            fig, ax = plt.subplots(1,3)
            img[img<0] = 0
            rec[rec<0] = 0
            target[target<0] = 0
            ax[0].imshow(img)
            ax[1].imshow(rec)
            ax[2].imshow(target)
            plt.savefig(os.path.join(args.time_path,'image_reconstruct_{}_{}.jpg'.format(epoch,num)))
            plt.close()



def main(args):

    Epoches = args.epochs
    Batch_Size = args.batch_size
    gpu = args.gpu
    filename = args.model_file
    LR = args.learning_rate
    pretrained = args.pretrained
    workers = args.workers

    logging.info("Batch_Size: {}".format(Batch_Size))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = ('cuda:{}'.format(gpu[0]) if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if len(gpu) > 1:
        device_name = 'cuda'
    logging.info("--------------using {}----------------".format(device_name))
    device = torch.device(device_name)

    
    training_data = ImageFileDataset(args.data_path, train=True)
    val_data = ImageFileDataset(args.data_path, train=False)
    sz = training_data.get_size()

    train_dataloader = DataLoader(training_data, batch_size=Batch_Size, shuffle=False, persistent_workers = True, prefetch_factor = 4, 
                                  drop_last=True,num_workers=workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False,
                                drop_last=False,num_workers=workers, pin_memory=True)
    test_dataloader = DataLoader(val_data, batch_size=1, shuffle=False,
                                 drop_last=False)
 
    # 3.模型加载, 并对模型进行微调
    net = LADMM(mode=args.mode,psf_file=args.psf_file, iter= args.layer_num,senor_size =[sz[1],sz[0]], filter= args.filter_num, ks=args.kernel_size)
    logging.info(net)

    if pretrained is not None:
        dict = torch.load(pretrained)
        net.load_state_dict(dict["state_dict"])

    # 4.pytorch fine tune 微调(冻结一部分层)。这里是冻结网络前30层参数进行训练。

    net.to(device)
 
    net = torch.nn.DataParallel(net, device_ids = gpu)

    # dummy_input = torch.rand(8, 9)  # 网络中输入的数据维度
    # target  = torch.rand(8, 282)  # 网络中输入的数据维度
    # with SummaryWriter(comment='ADMM') as w:
    #     w.add_graph(net, (dummy_input,target))  # net是你的网络名

    # 5.定义损失函数，以及优化器
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.L1Loss(reduction='sum')
    criterions = [torch.nn.MSELoss(reduction='mean'), lpips.LPIPS(net='alex',lpips=False).to(device)]

    optimizer = optim.Adam(net.parameters(), lr=LR)    
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    best_acc = torch.inf
    logging.info("Epoches:  {}".format(Epoches))
    scheduler = MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(args.tb_name)  # 存放log文件的目录
    for epoch in range(Epoches):
        # setup_seed(20)
        logging.info("learning ratio: {}".format(scheduler.get_last_lr()))
        # print(optimizer.param_groups)
        train(train_dataloader, net, criterions, optimizer, epoch, device, args, writer)
        loss = val(val_dataloader, net, criterions, epoch, device, writer)
        scheduler.step()    
        # remember best acc@1 and save checkpoint
        is_best = loss < best_acc
        best_acc = min(loss, best_acc)

        state_dict = {'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
            }
        test(test_dataloader, net, device, epoch, args)
        if is_best:
            # torch.save(state_dict, filename)
            save_file = os.path.join(args.time_path, args.model_file.split("/")[-1])
            torch.save(state_dict, save_file)
    writer.close()   
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model_file', default="best.pth", type=str, 
                        help='save model file path')
    parser.add_argument('--psf_file', default="data/psf.tiff", type=str, 
                        help='otf file path')
    parser.add_argument('--data_path', default="data/SpectralResponse_9_1024", type=str, 
                        help='data_path, size: 3380*9')   
    parser.add_argument('--mode', default="l1+tv", type=str, 
                        help='mode only_l1, only_tv, l1_tv, l1_cnn')   
    parser.add_argument('--multiple', default=2, type=int, 
                        help='multiple of chan')
    parser.add_argument('--layer_num', default=9, type=int, 
                        help='layer num of ista')    
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--steps', nargs='+', default=[50, 70], type=int, 
                        help='Learning rate step')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='use pre-trained model')
    parser.add_argument('--gpu', nargs='+', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--log_file', default="file.log", type=str,
                        help='log file path.')
    parser.add_argument('--have_noise', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='have_noise') 
    parser.add_argument('--sig_min', default=200, type=int, help='low value sigma of gauss line shape')  
    parser.add_argument('--sig_max', default=500, type=int, help='high value sigma of gauss line shape')  


    parser.add_argument('--filter_num', default=32, type=int, help='num of filter for cnn regular')  
    parser.add_argument('--kernel_size', default=3, type=int, help='kernel_size for cnn regular')      
    args = parser.parse_args()


    now = datetime.datetime.now()
    time_name = now.strftime("%Y%m%d%H%M%S")
    args.time_name = time_name


    time_path = os.path.join("debug", time_name)
    if not os.path.exists(time_path):
        os.mkdir(time_path)
    args.time_path = time_path
    logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s',level=logging.INFO,
                        handlers=[logging.FileHandler(args.log_file,mode="w"), logging.StreamHandler(),
                                  logging.FileHandler("{}/{}".format(time_path, args.log_file.split("/")[-1]),mode="w")])
    logging.info("{}".format(vars(args)))
    args.tb_name = os.path.join(args.time_path) #"runs/logs_{}".format(time_name)
    main(args)


    
