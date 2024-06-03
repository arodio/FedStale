# Tmp

**First output of generate_data:**

(venv_fedstale) C:\Users\charlotte.rodriguez\Downloads\gitlab-REPOSITORIES\FedStale\data\mnist>python generate_data.py --n_tasks 24 --s_frac 0.2 --test_tasks_frac 0.0 --seed 12345
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to raw_data/MNIST\raw\train-images-idx3-ubyte.gz
100.0%
Extracting raw_data/MNIST\raw\train-images-idx3-ubyte.gz to raw_data/MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to raw_data/MNIST\raw\train-labels-idx1-ubyte.gz
100.0%
Extracting raw_data/MNIST\raw\train-labels-idx1-ubyte.gz to raw_data/MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to raw_data/MNIST\raw\t10k-images-idx3-ubyte.gz
100.0%
Extracting raw_data/MNIST\raw\t10k-images-idx3-ubyte.gz to raw_data/MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to raw_data/MNIST\raw\t10k-labels-idx1-ubyte.gz
100.0%
Extracting raw_data/MNIST\raw\t10k-labels-idx1-ubyte.gz to raw_data/MNIST\raw

## For the new experiments

**For the new experiement (cross silo with CI based availability):**
```bash
cd data/mnist
python generate_data.py --n_tasks 6 --s_frac 0.2 --test_tasks_frac 0.0 --seed 12345
cd ../..
python train.py ...
```

## Requirements

**For requirements.txt file:**
```
numpy
--extra-index-url https://download.pytorch.org/whl/cu121
torch torchvision
scikit-learn
tqdm
tensorboard
tensorflow
```