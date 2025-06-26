import os
import json
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# =========================
# 1. Preparar labels2index.json si no existe
# =========================
labels2index_path = 'CIFAR-10/labels2index.json'

if not os.path.exists(labels2index_path):
    df = pd.read_csv('CIFAR-10/training_labels.csv')
    clases = sorted(df['label'].unique())
    labels2index = {label: idx for idx, label in enumerate(clases)}
    with open(labels2index_path, 'w') as f:
        json.dump(labels2index, f)
    print("✅ labels2index.json creado")
else:
    with open(labels2index_path) as f:
        labels2index = json.load(f)
    print("✅ labels2index.json cargado")

# =========================
# 2. Dividir dataset en 50% train / 50% val si no existen
# =========================
train_csv = 'CIFAR-10/train_labels.csv'
val_csv = 'CIFAR-10/val_labels.csv'
test_csv = 'CIFAR-10/test_labels.csv'  # asumimos existe

if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
    df = pd.read_csv('CIFAR-10/training_labels.csv')
    train_df, val_df = train_test_split(df, test_size=0.5, stratify=df['label'], random_state=42)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print("✅ train_labels.csv y val_labels.csv creados")
else:
    print("✅ train_labels.csv y val_labels.csv ya existen")

# =========================
# 3. Dataset personalizado para imágenes planas (para MLP) y normales (para CNN)
# =========================
class CIFAR10Dataset(Dataset):
    def __init__(self, images_folder, csv_labels_path, label_to_idx, transform=None, flatten=False):
        self.images_folder = images_folder
        self.labels_df = pd.read_csv(csv_labels_path)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.flatten = flatten

    def __getitem__(self, index):
        row = self.labels_df.loc[index]
        img_path = os.path.join(self.images_folder, row['image_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        if self.flatten:
            image = image.view(-1) if torch.is_tensor(image) else torch.tensor(image).reshape(-1)
        label = self.label_to_idx[row['label']]
        return image, label

    def __len__(self):
        return len(self.labels_df)

# =========================
# 4. Definir modelos SimpleCNN y SimpleMLP
# =========================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16
            nn.Conv2d(32, 64, 3, padding=1),  # 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_size=3072, hidden_size=256, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

# =========================
# 5. Función de evaluación
# =========================
def evaluar(modelo, data_loader, loss_fn, device):
    modelo.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = modelo(x)
            loss = loss_fn(outputs, y)
            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item() * x.size(0)
            total_correct += torch.sum(preds == y).item()
    avg_loss = total_loss / len(data_loader.dataset)
    acc = total_correct / len(data_loader.dataset)
    return avg_loss, acc

# =========================
# 6. Entrenar un modelo PyTorch (función general)
# =========================
def entrenar_modelo(modelo, train_loader, val_loader, loss_fn, optimizer, device, num_epochs, nombre_guardado):
    mejor_acc = 0
    metrics = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    modelo.to(device)

    for epoch in range(num_epochs):
        modelo.train()
        running_loss = 0
        running_correct = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = modelo(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * x.size(0)
            running_correct += torch.sum(preds == y).item()
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_correct / len(train_loader.dataset)
        val_loss, val_acc = evaluar(modelo, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        if val_acc > mejor_acc:
            mejor_acc = val_acc
            torch.save(modelo.state_dict(), nombre_guardado)
            print(f"📦 Mejor modelo guardado en {nombre_guardado}")

    # Guardar métricas train/val
    with open(nombre_guardado.replace(".pth", "_metrics.json"), "w") as f:
        json.dump(metrics, f)
    print(f"✅ Métricas train y val guardadas para {nombre_guardado}")

# =========================
# 7. Preparar device, parámetros y transforms
# =========================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

batch_size = 128
num_epochs = 10

transform_cnn = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

transform_mlp = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# =========================
# 8. Cargar datasets y dataloaders
# =========================
train_dataset_cnn = CIFAR10Dataset(
    images_folder='CIFAR-10/Train',
    csv_labels_path=train_csv,
    label_to_idx=labels2index,
    transform=transform_cnn,
    flatten=False
)
val_dataset_cnn = CIFAR10Dataset(
    images_folder='CIFAR-10/Train',
    csv_labels_path=val_csv,
    label_to_idx=labels2index,
    transform=transform_cnn,
    flatten=False
)
test_dataset_cnn = CIFAR10Dataset(
    images_folder='CIFAR-10/Test',
    csv_labels_path=test_csv,
    label_to_idx=labels2index,
    transform=transform_cnn,
    flatten=False
)

train_dataset_mlp = CIFAR10Dataset(
    images_folder='CIFAR-10/Train',
    csv_labels_path=train_csv,
    label_to_idx=labels2index,
    transform=transform_mlp,
    flatten=True
)
val_dataset_mlp = CIFAR10Dataset(
    images_folder='CIFAR-10/Train',
    csv_labels_path=val_csv,
    label_to_idx=labels2index,
    transform=transform_mlp,
    flatten=True
)
test_dataset_mlp = CIFAR10Dataset(
    images_folder='CIFAR-10/Test',
    csv_labels_path=test_csv,
    label_to_idx=labels2index,
    transform=transform_mlp,
    flatten=True
)

train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True)
val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=64)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=64)

train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=batch_size, shuffle=True)
val_loader_mlp = DataLoader(val_dataset_mlp, batch_size=64)
test_loader_mlp = DataLoader(test_dataset_mlp, batch_size=64)

# =========================
# 9. Entrenar SimpleCNN con SGD
# =========================
print("=== Entrenando SimpleCNN con SGD ===")
modelo_cnn_sgd = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer_sgd = optim.SGD(modelo_cnn_sgd.parameters(), lr=0.01, momentum=0.9)
entrenar_modelo(modelo_cnn_sgd, train_loader_cnn, val_loader_cnn, loss_fn, optimizer_sgd, device, num_epochs, "mejor_modelo_cnn_sgd.pth")

# =========================
# 10. Entrenar SimpleCNN con Adam
# =========================
print("=== Entrenando SimpleCNN con Adam ===")
modelo_cnn_adam = SimpleCNN().to(device)
optimizer_adam = optim.Adam(modelo_cnn_adam.parameters(), lr=0.001)
entrenar_modelo(modelo_cnn_adam, train_loader_cnn, val_loader_cnn, loss_fn, optimizer_adam, device, num_epochs, "mejor_modelo_cnn_adam.pth")

# =========================
# 11. Entrenar SimpleMLP con Adam
# =========================
print("=== Entrenando SimpleMLP con Adam ===")
modelo_mlp = SimpleMLP().to(device)
optimizer_mlp = optim.Adam(modelo_mlp.parameters(), lr=0.001)
entrenar_modelo(modelo_mlp, train_loader_mlp, val_loader_mlp, loss_fn, optimizer_mlp, device, num_epochs, "mejor_modelo_mlp.pth")

# =========================
# 12. Transfer Learning con ResNet-50
# =========================
print("=== Entrenando ResNet-50 (Transfer Learning) ===")
resnet = models.resnet50(pretrained=True)
# Modificar última capa para 10 clases CIFAR-10
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)
resnet = resnet.to(device)

optimizer_resnet = optim.Adam(resnet.parameters(), lr=0.0001)

# Usamos dataloaders con transform_cnn (resnet espera 224x224, pero para simplicidad usamos 32x32)
# Idealmente ajustar transform para 224x224, pero aquí simplificamos

entrenar_modelo(resnet, train_loader_cnn, val_loader_cnn, loss_fn, optimizer_resnet, device, num_epochs, "mejor_modelo_resnet50.pth")

# =========================
# 13. Evaluar test para PyTorch models y guardar métricas test
# =========================
def evaluar_y_guardar_test(modelo, test_loader, loss_fn, device, nombre_modelo):
    print(f"🔄 Evaluando mejor modelo guardado para {nombre_modelo} en test...")
    modelo.eval()
    loss, acc = evaluar(modelo, test_loader, loss_fn, device)
    print(f"📊 {nombre_modelo} Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")
    test_metrics = {"test_loss": loss, "test_acc": acc}
    with open(f"metrics_{nombre_modelo.lower()}_test_final.json", "w") as f:
        json.dump(test_metrics, f)
    print(f"✅ Métricas test guardadas en metrics_{nombre_modelo.lower()}_test_final.json")

# Cargar y evaluar mejor modelo guardado para cada uno
for model_name, path, test_loader in [
    ("cnn_sgd", "mejor_modelo_cnn_sgd.pth", test_loader_cnn),
    ("cnn_adam", "mejor_modelo_cnn_adam.pth", test_loader_cnn),
    ("mlp", "mejor_modelo_mlp.pth", test_loader_mlp),
    ("resnet50", "mejor_modelo_resnet50.pth", test_loader_cnn)
]:
    modelo = None
    if model_name == "cnn_sgd" or model_name == "cnn_adam":
        modelo = SimpleCNN()
    elif model_name == "mlp":
        modelo = SimpleMLP()
    elif model_name == "resnet50":
        modelo = models.resnet50(pretrained=False)
        num_ftrs = modelo.fc.in_features
        modelo.fc = nn.Linear(num_ftrs, 10)
    modelo.load_state_dict(torch.load(path))
    modelo = modelo.to(device)
    evaluar_y_guardar_test(modelo, test_loader, loss_fn, device, model_name)

# =========================
# 14. Entrenar y guardar métricas para SVM y KNN (modelos clásicos)
# =========================

# Función para cargar datos en numpy para sklearn
def cargar_numpy_dataset(csv_path, images_folder, label_to_idx):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    for _, row in df.iterrows():
        img_path = os.path.join(images_folder, row['image_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        img = img.flatten() / 255.0  # Normalizar
        X.append(img)
        y.append(label_to_idx[row['label']])
    X = np.array(X)
    y = np.array(y)
    return X, y

print("=== Entrenando y evaluando SVM ===")
X_train_svm, y_train_svm = cargar_numpy_dataset(train_csv, 'CIFAR-10/Train', labels2index)
X_val_svm, y_val_svm = cargar_numpy_dataset(val_csv, 'CIFAR-10/Train', labels2index)
X_test_svm, y_test_svm = cargar_numpy_dataset(test_csv, 'CIFAR-10/Test', labels2index)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_svm, y_train_svm)

train_acc_svm = accuracy_score(y_train_svm, svm.predict(X_train_svm))
val_acc_svm = accuracy_score(y_val_svm, svm.predict(X_val_svm))
test_acc_svm = accuracy_score(y_test_svm, svm.predict(X_test_svm))

metrics_svm = {
    "train_acc": train_acc_svm,
    "val_acc": val_acc_svm,
    "test_acc": test_acc_svm
}
with open("metrics_svm.json", "w") as f:
    json.dump(metrics_svm, f)
print(f"✅ Métricas SVM guardadas")

print("=== Entrenando y evaluando KNN ===")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_svm, y_train_svm)

train_acc_knn = accuracy_score(y_train_svm, knn.predict(X_train_svm))
val_acc_knn = accuracy_score(y_val_svm, knn.predict(X_val_svm))
test_acc_knn = accuracy_score(y_test_svm, knn.predict(X_test_svm))

metrics_knn = {
    "train_acc": train_acc_knn,
    "val_acc": val_acc_knn,
    "test_acc": test_acc_knn
}
with open("metrics_knn.json", "w") as f:
    json.dump(metrics_knn, f)
print(f"✅ Métricas KNN guardadas")

# =========================
# 15. Crear tabla comparativa con todas las métricas
# =========================

def cargar_metricas(nombre_json):
    with open(nombre_json) as f:
        data = json.load(f)
    train_acc = data.get("train_acc")
    val_acc = data.get("val_acc")
    test_acc = data.get("test_acc")
    return train_acc, val_acc, test_acc

modelos_metricas = {
    "SimpleCNN (SGD)": ("mejor_modelo_cnn_sgd_metrics.json", "mejor_modelo_cnn_sgd_metrics.json", "metrics_cnn_sgd_test_final.json"),
    "SimpleCNN (Adam)": ("mejor_modelo_cnn_adam_metrics.json", "mejor_modelo_cnn_adam_metrics.json", "metrics_cnn_adam_test_final.json"),
    "ResNet-50": ("mejor_modelo_resnet50_metrics.json", "mejor_modelo_resnet50_metrics.json", "metrics_resnet50_test_final.json"),
    "SimpleMLP": ("mejor_modelo_mlp_metrics.json", "mejor_modelo_mlp_metrics.json", "metrics_mlp_test_final.json"),
    "SVM": ("metrics_svm.json", "metrics_svm.json", "metrics_svm.json"),
    "KNN": ("metrics_knn.json", "metrics_knn.json", "metrics_knn.json")
}

tabla = []

for modelo, (train_file, val_file, test_file) in modelos_metricas.items():
    try:
        train_acc, _, _ = cargar_metricas(train_file)
        _, val_acc, _ = cargar_metricas(val_file)
        _, _, test_acc = cargar_metricas(test_file)
        tabla.append({
            "Modelo": modelo,
            "Train Accuracy": train_acc,
            "Validation Accuracy": val_acc,
            "Test Accuracy": test_acc
        })
    except Exception as e:
        print(f"Error al cargar métricas para {modelo}: {e}")

df_tabla = pd.DataFrame(tabla)
print("\n📋 Tabla comparativa de métricas:\n")
print(df_tabla)
