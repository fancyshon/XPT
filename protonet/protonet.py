import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import argparse
import random
from statistics import mean
from pathlib import Path
import copy

from easyfsl.datasets import CUB, MiniImageNet, TieredImageNet
from easyfsl.modules import resnet12
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate

import time
import datetime


random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def training_epoch(model_: nn.Module, data_loader: DataLoader, optimizer: Optimizer):
    all_loss = []
    model_.train()
    with tqdm(data_loader, total=len(data_loader), desc="Training") as tqdm_train:
        for images, labels in tqdm_train:
            optimizer.zero_grad()

            loss = LOSS_FUNCTION(model_(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)

def args_parser():

    parser = argparse.ArgumentParser(description="Train a model with classical training.")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="CUB")
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = args_parser()
    
    batch_size = args.batch_size
    n_workers = 12
    n_way = args.n_way
    n_shot = args.n_shot
    n_query = 10
    n_validation_tasks = 250
    n_epochs = args.n_epochs
    
    
    DEVICE = "cuda"


    if args.dataset == "CUB":
        train_set = CUB(split="train", training=True, image_size=224)
        val_set = CUB(split="val", training=False, image_size=224)
        test_set = CUB(split="test", training=False, image_size=224)
    elif args.dataset == "miniImageNet":
        train_set = MiniImageNet(root='datasets/MiniImageNet', split="train", training=True, image_size=224)
        val_set = MiniImageNet(root='datasets/MiniImageNet', split="val", training=False, image_size=224)
        test_set = MiniImageNet(root='datasets/MiniImageNet', split="test", training=False, image_size=224)
    elif args.dataset == "tieredImageNet":
        train_set = TieredImageNet(split="train", training=True, image_size=224)
        val_set = TieredImageNet(split="val", training=False, image_size=224)
        test_set = TieredImageNet(split="test", training=False, image_size=224)
        n_epochs = 100

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
        shuffle=True,
    )
    

    val_sampler = TaskSampler(
        val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )
    
    model = resnet12(
        use_fc=True,
        num_classes=len(set(train_set.get_labels())),
    ).to(DEVICE)

    few_shot_classifier = PrototypicalNetworks(model).to(DEVICE)
    
    # Calculate training time
    start_time = time.time()
    
    print("Dataset: ", args.dataset)
    print("Few shot task: ", n_way, "way", n_shot, "shot")
    
    # Print train_loader shape
    # print(f"Train_loader shape: {len(train_loader)}")
    # # Print the number of classes and the number of samples in train_loader
    # print(f"Number of classes: {train_set.number_of_classes()}")
    # print(f"Number of samples: {len(train_set)}")

    # # Print train_loader shape
    # print(f"val_loader shape: {len(val_loader)}")
    # # Print the number of classes and the number of samples in train_loader
    # print(f"Number of classes: {val_set.number_of_classes()}")
    # print(f"Number of samples: {len(val_set)}")

    # exit()

    LOSS_FUNCTION = nn.CrossEntropyLoss()

    scheduler_milestones = [150, 180]
    scheduler_gamma = 0.1
    learning_rate = 1e-01
    tb_logs_dir = Path(".")

    train_optimizer = SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

    best_state = model.state_dict()
    best_validation_accuracy = 0.0
    validation_frequency = 10
    for epoch in range(n_epochs):
        average_loss = training_epoch(model, train_loader, train_optimizer)
        print(f"Epoch {epoch}, average loss : {average_loss}")

        if epoch % validation_frequency == validation_frequency - 1:

            # We use this very convenient method from EasyFSL's ResNet to specify
            # that the model shouldn't use its last fully connected layer during validation.
            model.set_use_fc(False)
            validation_accuracy = evaluate(
                few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
            )
            model.set_use_fc(True)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_state = copy.deepcopy(few_shot_classifier.state_dict())
                best_state = model.state_dict()
                # state_dict() returns a reference to the still evolving model's state so we deepcopy
                # https://pytorch.org/tutorials/beginner/saving_loading_models
                print("Ding ding ding! We found a new best model! 🏆")
                print("Accuracy: ", best_validation_accuracy)

            tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

        tb_writer.add_scalar("Train/loss", average_loss, epoch)

        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()

    model.load_state_dict(best_state)
    n_test_tasks = 1000

    test_sampler = TaskSampler(
        test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    
    train_end_time = time.time()


    # Second step: we instantiate a few-shot classifier using our trained ResNet as backbone, and run it on the test data. We keep using Prototypical Networks for consistence, but at this point you could basically use any few-shot classifier that takes no additional trainable parameters.
    # 
    # Like we did during validation, we need to tell our ResNet to not use its last fully connected layer.

    model.set_use_fc(False)

    accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE)
    total_end_time = time.time()
        
    train_time = datetime.timedelta(seconds=train_end_time - start_time)
    total_time = datetime.timedelta(seconds=total_end_time - start_time)

    print(f"Training time: {train_time}")
    print(f"Total time: {total_time}")
    print(f"Average accuracy : {(100 * accuracy):.2f} %")


