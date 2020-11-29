import torch
import numpy as np
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LABELS = ['anime', 'collecting', 'comics', 'computer games', 'dancing',
       'drawing', 'embroidery', 'fine arts', 'fitness', 'food',
       'gardening', 'man next to car', 'movies', 'pets', 'photography',
       'role playing', 'shopping', 'sport', 'tourism']

model = torch.load('ResNet18.innotechImageInterests.model')

def img_batch_handler(batch: list, device=DEVICE):
    inputs = torch.from_numpy(np.array(list(map(lambda img: transform(img).numpy(), batch))))
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs.argmax(-1).tolist()


def get_labels(predictions):
    for i, pred in enumerate(predictions):
        predictions[i] = LABELS[pred[0]]
    return predictions


def batch_top3(predicted_classes):
    d = {}
    for i in predicted_classes:
        d[i] = d.get(i, 0) + 1
    x: list = d.items()
    x.sort(key=lambda x: x[1])
    return list(map(lambda item: item[0], x))[:3]

# batch_top3(get_labels(img_batch_handler(batch))) batch - list of PIL Image objects
