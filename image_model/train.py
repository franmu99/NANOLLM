import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from image_model.model import ImageAttentionModel
from PIL import Image
import os

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=1e-4):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, _ = model(images)
            
            # Compute loss
            loss = criterion(reconstructed, images)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch.to(device)
                reconstructed, _ = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configurar transformaciones de datos
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Ruta a las imágenes
    image_dir = r"C:\Users\franm\OneDrive\Imágenes\michele_morrone"

    # Crear el conjunto de datos completo
    full_dataset = ImageDataset(image_dir, transform=transform)

    # Dividir el conjunto de datos en entrenamiento y validación
    train_size = int(0.8 * len(full_dataset))  # 80% para entrenamiento
    val_size = len(full_dataset) - train_size  # 20% para validación
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Crear los cargadores de datos
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Inicializar el modelo
    model = ImageAttentionModel(d_model=64, num_heads=8, num_layers=4)

    # Entrenar el modelo
    trained_model = train_model(model, train_loader, val_loader, num_epochs=50, device=device)

    # Guardar el modelo entrenado
    torch.save(trained_model.state_dict(), "image_attention_model.pth")