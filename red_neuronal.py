import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim

torch.manual_seed(123) #fijamos la semilla
trans = transforms.Compose([transforms.ToTensor()]) #Transformador para el dataset

root="./data"
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 128
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print ('Trainning batch number: {}'.format(len(train_loader)))
print ('Testing batch number: {}'.format(len(test_loader)))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)#capa oculta
        self.fc2 = nn.Linear(256, 10)#capa de salida
        self.loss_criterion = nn.CrossEntropyLoss()#Función de pérdida
        
    def forward(self, x, target):
        x = x.view(-1, 28*28)#transforma las imágenes de tamaño (n, 28, 28) a (n, 784)
        x = F.relu(self.fc1(x))#Función de activación relu en la salida de la capa oculta
        x = F.softmax(self.fc2(x), dim=1)#Función de activación softmax en la salida de la capa oculta
        loss = self.loss_criterion(x, target)#Calculo de la función de pérdida
        return x, loss


model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def evaluate(model, dataset_loader, optimizer, train=False):
    correct_cnt, ave_loss = 0, 0#Contador de aciertos y acumulador de la función de pérdida
    count = 0#Contador de muestras
    for batch_idx, (x, target) in enumerate(dataset_loader):
        count += len(x)#sumamos el tamaño de batch, esto es porque n_batches*tamaño_batch != n_muestras
        if train:
            optimizer.zero_grad()#iniciamos a 0 los valores de los gradiente
        x, target = Variable(x), Variable(target)#Convertimos el tensor a variable del modulo autograd
        score, loss = model(x, target)#realizamos el forward
        _, pred_label = torch.max(score.data, 1)#pasamos de one hot a número
        correct_cnt += (pred_label == target.data).sum()#calculamos el número de etiquetas correctas
        ave_loss += loss.item()#sumamos el resultado de la función de pérdida para mostrar después
        if train:
            loss.backward()#Calcula los gradientes y los propaga 
            optimizer.step()#adaptamos los pesos con los gradientes propagados
    accuracy = correct_cnt/count#Calculamos la precisión total
    ave_loss /= count#Calculamos la pérdida media
    print ('==>>>loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))#Mostramos resultados

for epoch in range(10):
	print("Epoch: {}".format(epoch))
	print("Train")
	evaluate(model, train_loader, optimizer, train=True)
	print("Test")
	evaluate(model, test_loader, optimizer, train=False)