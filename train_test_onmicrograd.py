import gzip
import random

from micrograd.multi_layer_perceptron import MLP


# Load mnist
def convert_to_np(imgf, labelf, n):
    f = gzip.open(imgf, "rb")
    l = gzip.open(labelf, "rb")

    f.read(16)
    l.read(8)
    input_image_data = []
    input_label_data = []

    for i in range(n):
        input_label_data.append(ord(l.read(1)))
        image = []
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        input_image_data.append(image)

    f.close()
    l.close()
    return input_image_data, input_label_data


train_data, train_label = convert_to_np(
    "./dataset/mnist/train-images-idx3-ubyte.gz",
    "./dataset/mnist/train-labels-idx1-ubyte.gz",
    20,
)
test_data, test_label = convert_to_np(
    "./dataset/mnist/t10k-images-idx3-ubyte.gz",
    "./dataset/mnist/t10k-labels-idx1-ubyte.gz",
    1,
)

zipped = list(zip(train_data, train_label))
random.shuffle(zipped)
train_data, train_label = zip(*zipped)
train_data, train_label = list(train_data), list(train_label)

print("Dataset loaded")


one_hot_dict = {
    0: [1,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0],
    8: [0,0,0,0,0,0,0,0,1,0],
    9: [0,0,0,0,0,0,0,0,0,1],
}



# Declare the network
n = MLP(2, [784, 10])

for _ in range(10):
    ypred = [n(x) for x in train_data]
    individual_loss = []
    for ygt, yout in zip(train_label,ypred):
        for youtt,ygtt in zip(yout,one_hot_dict[ygt]):
            individual_loss.append((youtt-ygtt)**2)


    loss = sum( individual_loss )
    print(loss)
    for p in n.parameters():
        p.grad = 0.0

    loss.backward()
    for p in n.parameters():
        p.data += -(0.1 * p.grad)

ypred = [n(x) for x in test_data]
for ygt, yout in zip(test_label, ypred):
    for youtt,ygtt in zip(yout,one_hot_dict[ygt]):
        print("Label: ",ygtt)
        print("Predictions: ",youtt)

