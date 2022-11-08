import numpy as np
from PIL import Image
import os

a = np.zeros((256,256))
print(a)

for i in range(256):
    k = np.full((256,),5*i)
    a[i] = k

a = a.T
print(a)
img = Image.fromarray(a).convert("L")

file_path = os.path.realpath(__file__)
#import pathlib
#file_path = pathlib.Path(__file__).resolve()
print(file_path)
save_path = os.path.join(os.path.dirname(file_path),"revolve2","experiments","multienvironment_experiment","environment_xml","terrain2.png")
img.save(save_path)
img.show()

