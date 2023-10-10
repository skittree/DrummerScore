from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import librosa
import subprocess as sp
import torchaudio
import torchvision
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np

drum_labels = ['kick', 'snare', 'hihat', 'tom', 'crash', 'ride']

class DrumCNN(nn.Module):
    def __init__(self):
        super(DrumCNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.4)
        
        self.fc3 = nn.Linear(256, len(drum_labels))
    
    def forward(self, x):
        # Convolutional Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
class DrumDataset(Dataset):
    def __init__(self, df, audio, transform, window_size=8192):
        self.df = df
        self.window_size = window_size
        self.transform = transform
        self.audio = audio
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load the onset time, label, and track name for the given index
        row = self.df.iloc[idx]
        onset_time = row['onset_time']
        labels = row[drum_labels].astype(int).values.flatten()
        labels = torch.tensor(labels).float()

        audio = self.audio[0]
        sr = self.audio[1]

        onset_window = audio[int(onset_time*sr)-self.window_size//2:int(onset_time*sr)+self.window_size//2]
        spec = self.transform(onset_window)
        return spec, labels
    
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: x.to("cuda")),
    torchvision.transforms.Lambda(lambda x: torch.stack([
            torchaudio.transforms.MelSpectrogram(
                n_fft=1024,
                hop_length=64,
                n_mels=128
            ).to("cuda")(x),
    torchaudio.transforms.MFCC(
                n_mfcc=128,
                melkwargs={'n_fft': 1024, 'hop_length': 64, 'n_mels': 128}).to("cuda")(x)
            ], dim=0).to("cuda")),
    torchvision.transforms.Lambda(lambda x: torch.stack([
            torchaudio.transforms.AmplitudeToDB().to("cuda")(x[0]),
            x[0],
            x[1]
        ], dim=0).to("cuda")),
    torchvision.transforms.Lambda(lambda x: x.to("cpu"))
])

def transcribe_drums(file):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(os.getcwd())
    checkpoint = torch.load('editor/models/model.pth')
    model = DrumCNN().to(device)
    model.load_state_dict(checkpoint['model'])

    # command = ["demucs", "--mp3", "--mp3-bitrate", "320", "--two-stems=drums", fc.selected]
    # if torch.cuda.is_available():
    #     print("Generating splits for \""+fc.selected_filename+"\" with GPU...")
    #     sp.run(command)

    # demucs_path = os.path.join("separated/htdemucs", fc.selected_filename[:-4], "drums.mp3")

    audio, sr = torchaudio.load(file, format="mp3")
    if audio.shape[0] == 2:
        audio = torch.mean(audio, dim=0, keepdim=False)

    y = audio.numpy()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=1024)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = pd.DataFrame(librosa.frames_to_time(onset_frames, sr=sr), columns=['onset_time'])
    onset_times[drum_labels] = False

    pred_dataset = DrumDataset(onset_times, (audio, sr), transforms)
    pred_loader = DataLoader(pred_dataset, batch_size=16, shuffle=False)

    model.eval()
    predicted_labels=[]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(pred_loader, total=len(pred_loader), unit='batch', desc=f"Labeling")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted_labels.extend((outputs>0.0).cpu().numpy().tolist())
    onset_times[drum_labels] = predicted_labels
    return onset_times