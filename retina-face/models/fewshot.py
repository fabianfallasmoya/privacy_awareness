import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.models import resnet18
from tqdm import tqdm

from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average

N_WAY = 1  # Number of classes in a task
N_SHOT = 100  # Number of images per class in the support set
N_QUERY = 1  # Number of images per class in the query set
N_EVALUATION_TASKS = 100

class CelebAFace(CelebA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Get the original image and attributes
        img, target = super().__getitem__(index)

        # Override the target to a single-class classification: "Face"
        # Here we use 1 as the label for all images since they are all faces
        target = 1

        return img, target

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.support_images = None
        self.support_images = None

    def forward(
        self,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(self.support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(self.support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(self.support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


def create_model():
    image_size = 128

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define the root directory where CelebA is stored
    root = "./data"

    #train_set = CelebAFace(root=root, split="train", transform=transform, download=False)
    test_set = CelebAFace(root=root, split="test", transform=transform, download=False)

    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    print(convolutional_network)

    model = PrototypicalNetworks(convolutional_network).cuda()

    # The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
    test_set.get_labels = lambda: [
        instance[1] for instance in test_set
    ]
    test_sampler = TaskSampler(
        test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
    )

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    
    (
        example_support_images,
        example_support_labels,
        example_query_images,
        example_query_labels,
        example_class_ids,
    ) = next(iter(test_loader))

    model.support_images = example_support_images.cuda()
    model.support_labels = example_support_labels.cuda()

    model.eval()
    return model