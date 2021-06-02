import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec

import horovod.torch as hvd
import os
import math
import glob
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-recs', default=os.path.expanduser('~/imagenet/train/*'),
                    help='path to training data')
parser.add_argument('--train-idx', default=os.path.expanduser('~/imagenet/idx_files/train/*'),
                    help='path to training indeces')
parser.add_argument('--val-recs', default=os.path.expanduser('~/imagenet/validation/*'),
                    help='path to validation data')
parser.add_argument('--val-idx', default=os.path.expanduser('~/imagenet/idx_files/validation/*'),
                    help='path to validation indeces')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')


def dali_dataloader(
        tfrec_filenames,
        tfrec_idx_filenames,
        shard_id=0,
        num_shards=1,
        batch_size=128,
        num_threads=4,
        resize=256,
        crop=224,
        prefetch=2,
        training=True,
        gpu_aug=False,
        cuda=True):
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=torch.cuda.current_device() if cuda else None)
    with pipe:
        inputs = fn.readers.tfrecord(
            path=tfrec_filenames,
            index_path=tfrec_idx_filenames,
            random_shuffle=training,
            shard_id=shard_id,
            num_shards=num_shards,
            initial_fill=10000,
            read_ahead=True,
            prefetch_queue_depth=prefetch,
            name='Reader',
            features={
                'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
            })
        jpegs = inputs["image/encoded"]
        if training:
            # decode jpeg and random crop
            images = fn.decoders.image_random_crop(jpegs,
                device='mixed' if gpu_aug else 'cpu',
                output_type=types.RGB,
                random_aspect_ratio=[crop/resize, resize/crop],
                random_area=[crop/resize, 1.0],
                num_attempts=100)
            images = fn.resize(images,
                            device='gpu' if gpu_aug else 'cpu',
                            resize_x=resize,
                            resize_y=resize,
                            dtype=types.FLOAT,
                            interp_type=types.INTERP_TRIANGULAR)
            flip_lr = fn.random.coin_flip(probability=0.5)

            # additional training transforms
            images = fn.rotate(images, angle=fn.random.uniform(range=(-30, 30)),
                                keep_size=True, fill_value=0)
            # ... https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html
        else:
            # decode jpeg and resize
            images = fn.decoders.image(jpegs,
                                    device='mixed' if gpu_aug else 'cpu',
                                    output_type=types.RGB)
            images = fn.resize(images,
                            device='gpu' if gpu_aug else 'cpu',
                            resize_shorter=resize,
                            dtype=types.FLOAT,
                            interp_type=types.INTERP_TRIANGULAR)
            flip_lr = False

        # center crop and normalise
        images = fn.crop_mirror_normalize(images,
            dtype=types.FLOAT,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=flip_lr)
        label = inputs["image/class/label"] - 1  # 0-999
        label = fn.element_extract(label, element_map=0)  # Flatten
        if cuda:  # transfer data to gpu
            pipe.set_outputs(images.gpu(), label.gpu())
        else:
            pipe.set_outputs(images, label)

    pipe.build()
    loader = PyTorchIterator(
        pipe, reader_name="Reader",
        auto_reset=True,
        last_batch_policy='DROP' if training else 'PARTIAL')
    return loader


def train(epoch):
    model.train()
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, ) in enumerate(train_loader):
            data, target = data['data'], data['label']
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for (data, ) in val_loader:
                data, target = data['data'], data['label']
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    train_loader = dali_dataloader(
        sorted(glob.glob(args.train_recs)),
        sorted(glob.glob(args.train_idx)),
        batch_size=allreduce_batch_size,
        shard_id=hvd.rank(),
        num_shards=hvd.size(),
        training=True,
        cuda=args.cuda
    )

    val_loader = dali_dataloader(
        sorted(glob.glob(args.val_recs)),
        sorted(glob.glob(args.val_idx)),
        batch_size=allreduce_batch_size,
        shard_id=hvd.rank(),
        num_shards=hvd.size(),
        training=False,
        cuda=args.cuda
    )

    # Set up standard ResNet-50 model.
    model = models.resnet50()

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        validate(epoch)
        save_checkpoint(epoch)
