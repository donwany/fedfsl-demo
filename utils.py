from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot

image_size = 28


# NB: background=True selects the train set, background=False selects the test set
# It's the nomenclature from the original paper, we just have to deal with it

def load_data():
    train_set = Omniglot(root="./omnigplot", background=True,
                         transform=transforms.Compose(
                             [
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.RandomResizedCrop(image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]
                         ),
                         download=True,
                         )

    test_set = Omniglot(root="./omnigplot", background=False,
                        transform=transforms.Compose(
                            [
                                # Omniglot images have 1 channel, but our model will expect 3-channel images
                                transforms.Grayscale(num_output_channels=3),
                                transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                            ]
                        ),
                        download=True,
                        )

    N_WAY = 5  # Number of classes in a task
    N_SHOT = 5  # Number of images per class in the support set
    N_QUERY = 10  # Number of images per class in the query set

    N_TRAINING_EPISODES = 5
    N_EVALUATION_TASKS = 5

    train_set.get_labels = lambda: [instance[1] for instance in train_set._flat_character_images]
    test_set.get_labels  = lambda: [instance[1] for instance in test_set._flat_character_images]

    train_sampler = TaskSampler(
        train_set,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=N_TRAINING_EPISODES
    )

    test_sampler = TaskSampler(
        test_set,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=N_EVALUATION_TASKS
    )

    # test loader
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    # train loader
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    return train_loader, test_loader
