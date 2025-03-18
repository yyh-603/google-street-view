import torch
import torch.nn as nn
from torchvision import datasets, models
import torchvision.transforms as transforms
import torch.optim
import os
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ordered by city location, counter-clockwise
ORDERED_CITY = ["Keelung City", "New Taipei City", "Taipei City", "Taoyuan City", "Hsinchu County", "Hsinchu City", "Miaoli County", "Taichung City", "Changhua County", "Nantou County", "Yunlin County", "Chiayi County", "Chiayi City", "Tainan City", "Kaohsiung City", "Pingtung County", "Taitung County", "Hualien County", "Yilan County"]


if __name__ == '__main__':
    torch.manual_seed(114514)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(114514)

    # define the transformation
    transform = transforms.Compose([
        transforms.Resize((360, 360)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the data
    data_dir = './resnet_data'
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(114514))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # get the device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

    # change the class_to_idx
    train_dataset.class_to_idx = {key: value for value, key in enumerate(ORDERED_CITY)}
    val_dataset.class_to_idx = {key: value for value, key in enumerate(ORDERED_CITY)}
    test_dataset.class_to_idx = {key: value for value, key in enumerate(ORDERED_CITY)}

    for idx in range(len(train_dataset.samples)):
        _, class_idx = train_dataset.samples[idx]
        class_name = train_dataset.classes[class_idx]

        train_dataset.samples[idx] = (train_dataset.samples[idx][0], train_dataset.class_to_idx[class_name])
        train_dataset.targets[idx] = train_dataset.class_to_idx[class_name]

    for idx in range(len(val_dataset.samples)):
        _, class_idx = val_dataset.samples[idx]
        class_name = val_dataset.classes[class_idx]

        val_dataset.samples[idx] = (val_dataset.samples[idx][0], val_dataset.class_to_idx[class_name])
        val_dataset.targets[idx] = val_dataset.class_to_idx[class_name]

    for idx in range(len(test_dataset.samples)):
        _, class_idx = test_dataset.samples[idx]
        class_name = test_dataset.classes[class_idx]

        test_dataset.samples[idx] = (test_dataset.samples[idx][0], test_dataset.class_to_idx[class_name])
        test_dataset.targets[idx] = test_dataset.class_to_idx[class_name]

    # use existing model
    # uncomment this block if you want to use the existing model
    # resnet = models.resnet50()
    # resnet.fc = nn.Linear(resnet.fc.in_features, 19)
    # resnet.load_state_dict(torch.load('resnet50_pre.pth', map_location=device))
    # resnet = resnet.to(device)

    # use pretrained model
    # comment this block if you want to use the existing model
    resnet = models.resnet50(pretrained=True)

    # configure the model
    nums_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(nums_ftrs, 19)

    resnet = resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.00003)

    train_loss = []
    val_loss = []

    num_epochs = 15
    # train the model
    for epoch in range(num_epochs):
        resnet.train()
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())

        resnet.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)

            val_loss.append(loss.item())
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Validation Accuracy: {100 * correct / total:.2f}%')
        torch.save(resnet.state_dict(), f'resnet50_00003_{epoch + 1}.pth')

    # plot the loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_graph.png')

    # test the model
    resnet.eval()
    with torch.no_grad():
        all_labels = []
        all_predictions = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        print(f'Test Accuracy: {accuracy:.4f}')

        # plot confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.class_to_idx.keys(), yticklabels=train_dataset.class_to_idx.keys())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')