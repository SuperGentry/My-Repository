import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

from my_dataset import TestData_myself, Data_myself
from model_local import swin_base_patch4_window7_224 as create_model
from utils import train_one_epoch, each_epoch_feature, train_meta_two_class_epoch
from test import test
from models import Net


class TrainerConfig:
    """Configuration class for training parameters"""
    def __init__(self):
        self.num_classes = 2
        self.epochs = 8
        self.batch_size = 18
        self.lr = 0.0001
        self.step_size = 3
        self.gamma = 0.1
        self.data_path = "/data/flower_photos"
        self.train_list_path = "path/to/train.txt"
        self.dev_list_path = "path/to/dev.txt"
        self.test_list_path = "path/to/test.txt"
        self.weights = "./swin_base_patch4_window7_224.pth"
        self.freeze_layers = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"


def setup_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_data_loaders(train_path: str, batch_size: int, img_size: int = 224) -> DataLoader:
    """Create and return data loader for training"""
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = Data_myself(listroot=train_path, transform=data_transform)
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers')

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw
    )


def initialize_model(config: TrainerConfig) -> torch.nn.Module:
    """Initialize and configure the model"""
    model = create_model().to(config.device)
    
    if config.weights and os.path.exists(config.weights):
        weights_dict = torch.load(config.weights, map_location=config.device)
        
        # Remove classification head weights if present
        if "model" in weights_dict:
            for k in list(weights_dict["model"].keys()):
                if "head" in k:
                    del weights_dict["model"][k]
            model.load_state_dict(weights_dict["model"], strict=False)
        else:
            model.load_state_dict(weights_dict, strict=False)
    
    if config.freeze_layers:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad_(False)
            else:
                print(f"Training {name}")
    
    return model


def main(config: TrainerConfig) -> None:
    """Main training function"""
    setup_seed()
    
    # Create directories if they don't exist
    os.makedirs("./weights", exist_ok=True)
    
    # Initialize components
    train_loader = get_data_loaders(config.train_list_path, config.batch_size)
    model = initialize_model(config)
    tb_writer = SummaryWriter("Train_Logs")
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=5e-2
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.step_size, 
        gamma=config.gamma
    )
    
    # Training loop
    for epoch in range(config.epochs):
        scheduler.step()
        
        train_loss, train_acc, _ = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=config.device,
            epoch=epoch
        )
        
        # Log metrics
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        
        # Save model
        torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")


if __name__ == '__main__':
    config = TrainerConfig()
    main(config)