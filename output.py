import matplotlib.pyplot as plt
from skimage import io
import sys

input_image = io.imread(f'{sys.argv[1]}/input.png')
output_image = io.imread(f'{sys.argv[1]}/output.png')

f, ax = plt.subplots(nrows=1,ncols=2)

ax[0].imshow(input_image, cmap='gray')
ax[1].imshow(output_image, cmap='gray')

f.tight_layout()

plt.savefig("comparison.png",bbox_inches='tight')
