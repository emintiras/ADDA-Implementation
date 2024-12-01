import torch
import torch.optim as optim
import json
import logging
import os
from src.models import Encoder, Classifier, Discriminator
from src.train_source import train_source, test_source
from src.train_target import train_target, test_target
from src.data_loader import get_data_loaders

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("logs.log")
    ]
)

logger = logging.getLogger(__name__)

def main(load_pretrained=False):
    try:
        # Load parameters from JSON file
        with open('params.json', 'r') as f:
            params = json.load(f)
        logger.info("Parameters loaded successfully.")

        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(device)
        logger.info(f"Using device: {device}")

        # Get data loaders
        loaders = get_data_loaders(batch_size=params['batch_size'], device=device)
        source_trainset = loaders['mnist_train']
        source_testset = loaders['mnist_test']
        target_trainset = loaders['usps_train']
        target_testset = loaders['usps_test']
        logger.info("Data loaders initialized.")

        # Initialize models
        source_encoder = Encoder(input_dim=1, hidden_dim=params['hidden_dim'], dropout_prob=params['dropout']).to(device)
        target_encoder = Encoder(input_dim=1, hidden_dim=params['hidden_dim'], dropout_prob=params['dropout']).to(device)
        classifier = Classifier(input_dim=500, num_classes=params['num_classes'], c_hidden_dim=params['c_hidden_dim']).to(device)
        discriminator = Discriminator(input_dim=500, d_hidden_dim=params['d_hidden_dim']).to(device)
        logger.info("Models initialized.")

        # Load pretrained models if the flag is True
        if load_pretrained and os.path.exists('saved_models'):
            try:
                source_encoder = torch.jit.load('saved_models/best_source_encoder.pt')
                target_encoder = torch.jit.load('saved_models/target_encoder.pt')
                classifier = torch.jit.load('saved_models/best_classifier.pt')
                logger.info("Pretrained models loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load pretrained models: {str(e)}", exc_info=True)
                return 
        else:
            logger.info("Training models from scratch.")

            # Optimizer configurations
            opt_source = optim.Adam(
                list(source_encoder.parameters()) + list(classifier.parameters()), 
                lr=params['lr_source'], 
                betas=(params['beta1_source'], params['beta2_source']),
                weight_decay=params['source_weight_decay']
            )

            opt_target = optim.Adam(
                target_encoder.parameters(), 
                lr=params['lr_target'], 
                betas=(params['beta1_target'], params['beta2_target']),
                weight_decay=params['target_weight_decay']
            )

            opt_discriminator = optim.Adam(
                discriminator.parameters(), 
                lr=params['lr_discriminator'], 
                betas=(params['beta1_discriminator'], params['beta2_discriminator']),
                weight_decay=params['discriminator_weight_decay']
            )
            logger.info("Optimizers configured.")

            # Pre-train the source encoder and classifier
            logger.info("Starting source model pre-training...")
            source_encoder, classifier = train_source(
                encoder=source_encoder,
                classifier=classifier,
                source_train=source_trainset,
                source_test=source_testset,
                num_epochs=params['epochs_pretrain'],
                device=device,
                optimizer=opt_source
            )
            logger.info("Source model pre-training completed.")

            # Adapt target encoder using adversarial training
            logger.info("Starting adversarial adaptation of the target encoder...")
            target_encoder = train_target(
                source_encoder=source_encoder,
                target_encoder=target_encoder,
                discriminator=discriminator,
                source_train=source_trainset,
                target_train=target_trainset,
                num_epochs=params['epochs_adapt'],
                target_optimizer=opt_target,
                discriminator_optimizer=opt_discriminator,
                device=device
            )
            logger.info("Adversarial adaptation completed.")

        # Evaluate models on test sets
        logger.info("Evaluating source model...")
        source_acc = test_source(
            encoder=source_encoder,
            classifier=classifier,
            source_test=source_testset,
            device=device
        )
        logger.info(f"Source Test Accuracy: {source_acc:.2f}%")

        # Evaluate model on target dataset
        logger.info("Evaluating source model before adda...")
        source_acc_1 = test_source(
            encoder=source_encoder,
            classifier=classifier,
            source_test=target_testset,
            device=device
        )
        logger.info(f"Source Test Accuracy: {source_acc_1:.2f}%")

        logger.info("Evaluating target model after adda...")
        target_acc = test_target(
            encoder=target_encoder,
            classifier=classifier,
            data_loader=target_testset,
            device=device
        )
        logger.info(f"Target Test Accuracy: {target_acc:.2f}%")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    load_pretrained = True
    main(load_pretrained=load_pretrained)
