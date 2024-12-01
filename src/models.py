from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, stride=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_prob),
            nn.Flatten(),
            nn.Linear(in_features=hidden_dim * 4 * 4, out_features=500)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
        
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes,  c_hidden_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(in_features=input_dim, out_features=c_hidden_dim),
            # nn.ReLU(),
            nn.Linear(in_features=input_dim, out_features=num_classes)
        )
    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, d_hidden_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=d_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=input_dim, out_features=d_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=d_hidden_dim, out_features=2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)