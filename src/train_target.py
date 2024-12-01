import torch.nn as nn
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("logs.log") 
    ]
)

logger = logging.getLogger(__name__)

def train_target(source_encoder, target_encoder, discriminator, source_train, target_train, num_epochs, target_optimizer, discriminator_optimizer, device):
    try:
        logger.info("Starting target encoder training...")

        source_encoder, target_encoder, discriminator = (
            source_encoder.to(device), target_encoder.to(device), discriminator.to(device)
        )
        target_encoder.load_state_dict(source_encoder.state_dict())
        logger.debug("Loaded source encoder state into target encoder.")

        criterion_disc = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            epoch_loss_disc, epoch_loss_target, total_steps = 0.0, 0.0, 0
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            for step, ((source_data, _), (target_data, _)) in enumerate(zip(source_train, target_train)):
                total_steps += 1
                source_data, target_data = source_data.to(device), target_data.to(device)

                try:
                    # Train Discriminator
                    discriminator_optimizer.zero_grad()
                    discriminator.train()
                    target_encoder.eval()

                    source_features = source_encoder(source_data)
                    target_features = target_encoder(target_data)

                    source_pred = discriminator(source_features)
                    target_pred = discriminator(target_features)

                    source_labels = torch.ones(source_features.size(0), dtype=torch.long, device=device)
                    target_labels = torch.zeros(target_features.size(0), dtype=torch.long, device=device)

                    loss_disc = criterion_disc(source_pred, source_labels) + criterion_disc(target_pred, target_labels)
                    loss_disc.backward()
                    discriminator_optimizer.step()

                    epoch_loss_disc += loss_disc.item()
                    logger.debug(f"Step {step + 1}: Discriminator Loss = {loss_disc.item():.4f}")

                    # Train Target Encoder
                    target_optimizer.zero_grad()
                    discriminator.eval()
                    target_encoder.train()

                    target_features = target_encoder(target_data)
                    target_pred = discriminator(target_features)
                    target_labels = torch.ones(target_features.size(0), dtype=torch.long, device=device)

                    loss_target = criterion_disc(target_pred, target_labels)
                    loss_target.backward()
                    target_optimizer.step()

                    epoch_loss_target += loss_target.item()
                    logger.debug(f"Step {step + 1}: Target Encoder Loss = {loss_target.item():.4f}")

                except Exception as e:
                    logger.error(f"Error during step {step + 1}: {str(e)}")

            if total_steps > 0:
                epoch_loss_disc /= total_steps
                epoch_loss_target /= total_steps

            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"Discriminator Loss: {epoch_loss_disc:.4f}")
            logger.info(f"Target Encoder Loss: {epoch_loss_target:.4f}")

        best_discriminator = torch.jit.script(discriminator) 
        best_discriminator.save('saved_models/discriminator.pt')
        best_target_encoder = torch.jit.script(target_encoder) 
        best_target_encoder.save('saved_models/target_encoder.pt')
        logger.info("Training completed and models saved.")

    except Exception as e:
        logger.critical(f"Training failed: {str(e)}", exc_info=True)

    return target_encoder

def test_target(encoder, classifier, data_loader, device):
    try:
        logger.info("Starting target encoder evaluation...")
        encoder, classifier = encoder.to(device), classifier.to(device)
        encoder.eval()
        classifier.eval()

        total_loss, correct_predictions, total_samples = 0.0, 0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for step, (images, labels) in enumerate(data_loader):
                try:
                    images, labels = images.to(device), labels.to(device, dtype=torch.long)

                    target_features = encoder(images)
                    preds = classifier(target_features)

                    loss = criterion(preds, labels)
                    total_loss += loss.item() * images.size(0)

                    _, predictions = torch.max(preds, 1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_samples += labels.size(0)

                    logger.debug(f"Step {step + 1}: Batch Loss = {loss.item():.4f}")

                except Exception as e:
                    logger.error(f"Error during evaluation step {step + 1}: {str(e)}")

        if total_samples == 0:
            logger.warning("No samples found during evaluation.")
            return 0.0

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples * 100

        logger.info(f"Evaluation Summary: Average Loss = {average_loss:.4f}, Accuracy = {accuracy:.4f}")

        return accuracy

    except Exception as e:
        logger.critical(f"Evaluation failed: {str(e)}", exc_info=True)
        return 0.0
