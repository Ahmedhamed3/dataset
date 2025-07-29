import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class XrayLandmarkDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # scales to [0,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = Image.open(img_path).convert('L')
        orig_w, orig_h = img.size
        img = self.transform(img)

        coords = row[[
            'MF_x','MF_y',
            'Apex_of_PremolarMolar_x','Apex_of_PremolarMolar_y',
            'IAC_x','IAC_y',
            'LBM_90_x','LBM_90_y']].astype(float).values.astype('float32')
        norm = np.array([
            coords[0]/orig_w, coords[1]/orig_h,
            coords[2]/orig_w, coords[3]/orig_h,
            coords[4]/orig_w, coords[5]/orig_h,
            coords[6]/orig_w, coords[7]/orig_h
        ], dtype='float32')
        norm = np.clip(norm, 0.0, 1.0)
        return img, torch.from_numpy(norm)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*14*14, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = XrayLandmarkDataset('all_19_samples_landmarks_clean.csv', 'xray_images')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = SimpleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, 501):
        epoch_loss = 0.0
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        if epoch % 50 == 0 or epoch == 1:
            print(f'Epoch {epoch}/500 Loss: {epoch_loss/len(dataset):.4f}')

    torch.save(model.state_dict(), 'landmark_model.pth')
    return model, dataset

def evaluate(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds_list = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, gt = dataset[idx]
            input_img = img.unsqueeze(0).to(device)
            pred = model(input_img).cpu().squeeze(0).numpy()
            preds_list.append([dataset.df.loc[idx,'filename']] + list(pred))

            if idx == 0:
                # visualize first sample
                fig, ax = plt.subplots()
                ax.imshow(img.squeeze(), cmap='gray')
                gt_pts = gt.numpy() * 224
                pred_pts = pred * 224
                for i in range(0,8,2):
                    ax.scatter(gt_pts[i], gt_pts[i+1], c='g')
                    ax.scatter(pred_pts[i], pred_pts[i+1], c='r')
                ax.set_title('Green: GT, Red: Pred')
                plt.show()

    columns = ['filename','MF_x','MF_y','Apex_of_PremolarMolar_x','Apex_of_PremolarMolar_y','IAC_x','IAC_y','LBM_90_x','LBM_90_y']
    pd.DataFrame(preds_list, columns=columns).to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    model, dataset = train()
    evaluate(model, dataset)
