import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("logs.log",) 
    ]
)

logger = logging.getLogger(__name__)

def get_data_loaders(batch_size, device, data_root='data/', seed=42):
    """
    Returns data loaders for MNIST and USPS datasets.

    Args:
    batch_size (int): Batch size for the data loaders.
    data_root (str): Path to store/download datasets.
    seed (int): Seed for reproducibility.
    device (str): Device used for the generator ('cpu' or 'cuda').

    Returns:
    dict: Data loaders for train and test splits of MNIST and USPS.
    """
    try:
        logger.info("Initializing data loaders...")
        generator = torch.Generator(device=device).manual_seed(seed)

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        logger.info("Loading MNIST dataset...")
        mnist_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

        logger.info("Loading USPS dataset...")
        usps_train = datasets.USPS(root=data_root, train=True, download=True, transform=transform)
        usps_test = datasets.USPS(root=data_root, train=False, download=True, transform=transform)

        loaders = {
            'mnist_train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True, generator=generator),
            'mnist_test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False, generator=generator),
            'usps_train': DataLoader(usps_train, batch_size=batch_size, shuffle=True, generator=generator),
            'usps_test': DataLoader(usps_test, batch_size=batch_size, shuffle=False, generator=generator),
        }

        logger.info(f"Data loaders initialized successfully with batch size {batch_size}.")
        logger.debug(f"MNIST Train Size: {len(mnist_train)}, MNIST Test Size: {len(mnist_test)}")
        logger.debug(f"USPS Train Size: {len(usps_train)}, USPS Test Size: {len(usps_test)}")

        return loaders

    except Exception as e:
        logger.critical(f"Failed to initialize data loaders: {str(e)}", exc_info=True)
        raise
