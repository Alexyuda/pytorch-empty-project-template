from opts import parser
from datetime import datetime
import os
from glob import glob
from dataset import DummyDataset
from shutil import copyfile
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from model import DummyNet
from torchsummary import summary
from torch import optim
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def main(args):
    torch.manual_seed(0)

    now_data_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    log_dir = os.path.join(args.proj_dir, 'logs')
    checkpnt_base_dir = os.path.join(args.proj_dir, 'checkpnts')
    checkpnt_dir = os.path.join(checkpnt_base_dir, now_data_time)

    if not os.path.exists(checkpnt_base_dir):
        os.mkdir(checkpnt_base_dir)
    if not os.path.exists(checkpnt_dir):
        os.mkdir(checkpnt_dir)

    # copy src files
    src_files = glob(os.path.join(args.proj_dir, 'src', '*.py'))
    for src_file in src_files:
        copyfile(src_file, os.path.join(checkpnt_dir, os.path.split(src_file)[1]))

    ##################################### data #####################################

    dataset = DummyDataset(args=args)
    n_samples = len(dataset)
    train_indices = list(range(0, round(args.train_val_split_p * n_samples)))
    val_indices = list(range(round(args.train_val_split_p * n_samples), n_samples))
    train_sampler = SequentialSampler(train_indices)
    valid_sampler = SequentialSampler(val_indices)

    loader_train = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                               shuffle=False, pin_memory=True, num_workers=4)
    loader_val = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler,
                                             shuffle=False, pin_memory=True, num_workers=4)

    ##################################### training opts #####################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HomograhpyNet().to(device)
    summary(model=model, input_size=(3, args.win_size, args.win_size))

    if args.pre_load_net_fn:
        model.load_state_dict(
            torch.load(args.pre_load_net_fn, map_location=device)
        )
        print(f'Loaded net {args.load}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    writer = SummaryWriter(os.path.join(log_dir,
                                        now_data_time + f'_LR_{args.lr}_BS_{args.batch_size}'))
    print('run: tensorboard --logdir=' + log_dir + ' --host=127.0.0.1')

    ##################################### train #####################################
    for epoch in range(args.n_ephocs):
        epoch_loss = 0
        model.train()
        for img, bbox_gt in tqdm(loader_train, desc=f'training... Epoch {epoch}/{args.n_ephocs}'):
            img = img.to(device=device)
            gt = bbox_gt.to(device=device, dtype=torch.float32)
            pred = model(img)
            loss = criterion(bbox_pred, bbox_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader_train)
        print(f'train loss: {epoch_loss}')
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        torch.save(model.state_dict(),
                   os.path.join(checkpnt_dir, f'epoch{epoch}.pth'))

        ##################################### validation #####################################
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for img, bbox_gt in tqdm(loader_val, desc=f'validation... Epoch {epoch}/{args.n_ephocs}'):
                img = img.to(device=device)
                gt = bbox_gt.to(device=device)
                pred = model(img)
                loss = criterion(bbox_pred, bbox_gt)
                val_loss += loss.item()
            val_loss /= len(loader_val)
            print(f'val loss: {val_loss}')
            writer.add_scalar('Loss/eval', val_loss, epoch)

    writer.close()


if __name__ == "__main__":
    args_ = parser.parse_args()
    main(args_)
