import numpy as np
import gzip

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
        for j in range(28*28):
            image.append(ord(f.read(1)))
        input_image_data.append(image)

    f.close()
    l.close()
    return input_image_data, input_label_data

train_data, train_label = convert_to_np("./dataset/mnist/train-images-idx3-ubyte.gz", "./dataset/mnist/train-labels-idx1-ubyte.gz",60000)
test_data, test_label = convert_to_np("./dataset/mnist/t10k-images-idx3-ubyte.gz", "./dataset/mnist/t10k-labels-idx1-ubyte.gz",10000)
print(test_data)
print(test_label)