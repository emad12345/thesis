from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as transforms

writer = SummaryWriter("runs/test_image_log")

fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 2, 1])
ax.set_title("Test Plot")

buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = Image.open(buf)
image = transforms.ToTensor()(image)

writer.add_image("Sample_Plot", image, dataformats='CHW')
writer.close()
