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

def train_source(encoder, classifier, source_train, source_test, num_epochs, device, optimizer, patience=10):
    encoder = encoder.to(device)
    classifier = classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0
    epochs_without_improvement = 0
    epoch_losses = []

    try:
        for epoch in range(num_epochs):
            encoder.train()
            classifier.train()
            epoch_loss = 0
            epoch_steps = 0

            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] started.")

            for i, (images, labels) in enumerate(source_train):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                source_features = encoder(images)
                source_preds = classifier(source_features)

                loss = criterion(source_preds, labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                epoch_steps += labels.size(0)

            avg_loss = epoch_loss / epoch_steps
            epoch_losses.append(avg_loss)
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

            # Evaluate on the test set
            encoder.eval()
            classifier.eval()
            test_correct, test_total, test_loss = 0, 0, 0

            with torch.no_grad():
                for i, (images, labels) in enumerate(source_test):
                    images, labels = images.to(device), labels.to(device)
                    features = encoder(images)
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_acc = 100 * test_correct / test_total
            avg_test_loss = test_loss / test_total
            logger.info(f'Test Accuracy: {test_acc:.2f}%, Test Loss: {avg_test_loss:.4f}')

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                epochs_without_improvement = 0
                best_source_model = torch.jit.script(encoder)
                best_source_model.save('saved_models/best_source_encoder.pt')
                best_classifier = torch.jit.script(classifier)
                best_classifier.save('saved_models/best_classifier.pt')
                logger.info(f'New best model saved with test accuracy: {best_test_acc:.2f}%')
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.warning(f'Early stopping triggered after {patience} epochs without improvement.')
                break

        logger.info(f'Best test accuracy: {best_test_acc:.2f}%')

        encoder = torch.jit.load('saved_models/best_source_encoder.pt')
        classifier = torch.jit.load('saved_models/best_classifier.pt')

    except Exception as e:
        logger.critical(f"An error occurred during training: {str(e)}", exc_info=True)

    return encoder, classifier

def test_source(encoder, classifier, source_test, device):
    encoder = encoder.float().to(device).eval()
    classifier = classifier.float().to(device).eval()

    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    try:
        with torch.no_grad():
            for step, (images, labels) in enumerate(source_test):
                images, labels = images.to(device), labels.to(device, dtype=torch.long)

                source_features = encoder(images)
                source_preds = classifier(source_features)

                loss = criterion(source_preds, labels)
                total_loss += loss.item() * images.size(0)

                _, predictions = torch.max(source_preds, 1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples * 100

        logger.info(f'Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

        return accuracy

    except Exception as e:
        logger.critical(f"An error occurred during evaluation: {str(e)}", exc_info=True)
        return 0.0
