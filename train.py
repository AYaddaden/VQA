import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from data_loader import VQADataset
from model import VQAModel

from torch.optim import Adam


batch_size = 10
path = "/content/drive/MyDrive/boolean_answers_dataset_100"
image_folder = "boolean_answers_dataset_images_100"
descriptor = "boolean_answers_dataset_100.csv"

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

vqa_dataset = VQADataset(path, descriptor, image_folder, transform=transform)
dataset_length = vqa_dataset.size
train_size = int(dataset_length * 0.8)
val_size = int(dataset_length * 0.2)

train_dataset, val_dataset = random_split(vqa_dataset, [train_size, val_size])

vqa_dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
vqa_dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


embedding_size = 512
hidden_size_ch = 512
d_out = 2
dropout = 0.5
learning_rate = 1e-4
n_epochs = 5

vqa_model = VQAModel(embedding_size, hidden_size_ch, d_out=d_out, dropout=dropout, device=device)

initial_hidden = (
torch.zeros(1, batch_size, embedding_size).to(device), torch.zeros(1, batch_size, embedding_size).to(device))
vqa_model.to(device)


def map_answer(answer):
    ans = 1 if answer == 'yes' else 0
    return ans


criterion = torch.nn.CrossEntropyLoss()

optimizer = Adam(vqa_model.parameters(), lr=learning_rate)
for epoch in range(n_epochs):
    vqa_model.train()
    for batch_id, batch in enumerate(vqa_dataloader_train):
        image, question, answer = batch
        answer = torch.tensor(list(map(map_answer, answer)))

        image, answer = image.to(device), answer.to(device)

        output = vqa_model(image, question, initial_hidden)

        loss = criterion(output, answer)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print("loss value : {}\n".format(loss.item()))

    vqa_model.eval()
    total, correct = 0, 0
    for batch_id, batch in enumerate(vqa_dataloader_val):
        image, question, answer = batch
        answer = torch.tensor(list(map(map_answer, answer)))

        image, answer = image.to(device), answer.to(device)

        output = vqa_model(image, question, initial_hidden)

        sf_output = torch.nn.Softmax(dim=1)(output)  # softmax to obtain the probability distribution
        _, predicted = torch.max(sf_output, 1)  # decision rule, we select the max

        total += answer.size(0)
        correct += (predicted == answer).sum().item()

    print("[validation] accuracy: {:.3f}%\n".format(100 * correct / total))

torch.save({
    "model": vqa_model.state_dict()
}, 'vqa_model.pt')
