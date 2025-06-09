import compress
from compress import Tucker
from compress.sparse import SparseTensor

import model_compression
import model_compression.rieman_model
import model_compression.rieman_optimizer.tucker_optim

if __name__ == "__main__":
    print('all imports are working')