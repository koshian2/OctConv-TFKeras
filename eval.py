from models import *
import numpy as np
import time
from tqdm import tqdm

def eval_all():
    X = np.random.rand(50000, 32, 32, 3).astype(np.float32)
    for alpha in [0, 0.25, 0.5, 0.75]:
        if alpha == 0:
            model = create_normal_wide_resnet()
        else:
            model = create_octconv_wide_resnet(alpha)
        results = []
        for i in tqdm(range(20)):
            st = time.time()
            model.predict(X, batch_size=128)
            results.append(time.time()-st)
        results = np.array(results)
        print("alpha = ", alpha)
        print(f"Mean = {np.mean(results):.04}, Median = {np.median(results):.04}"+
              f", SD = {np.std(results):.04}")

if __name__ == "__main__":
    eval_all()
