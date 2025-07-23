import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from dataset import CardiacSegmentationDataset
from dataset import CardiacSegmentation2DDataset

def train(model, dataloader, loss_fn, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")

def main():
    # Paths
    image_dir = 'data/out/images'
    label_dir = 'data/out/labels'

    # Dataset & DataLoader
    dataset = CardiacSegmentation2DDataset(image_dir, label_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3
    )

    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    train(model, dataloader, loss_fn, optimizer, device, num_epochs=10)

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/unet_rv_segmentation.pth")
    print("Training complete. Model saved.")


if __name__ == '__main__':
    main()

