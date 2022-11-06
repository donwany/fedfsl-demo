import torch
from easyfsl.utils import sliding_average
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def fit(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        net
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    optimizer.zero_grad()
    classification_scores = net(
        support_images, support_labels, query_images
    )

    loss = criterion(classification_scores, query_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


# Train the model
def train_fedfsl(net, train_loader: DataLoader):
    log_update_frequency = 10

    all_loss = []
    net.train()

    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
        ) in tqdm_train:
            loss_value = fit(
                net, support_images, support_labels, query_images, query_labels
            )

            all_loss.append(loss_value)

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(
                    loss=sliding_average(all_loss, log_update_frequency)
                )


def evaluate_on_one_task(
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        net
):
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
                   torch.max(
                       net(support_images, support_labels, query_images)
                       .detach()
                       .data,
                       1,
                   )[1]
                   == query_labels
           ).sum().item(), len(query_labels)


def test_fedfsl(net, test_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    net.eval()
    with torch.no_grad():
        for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                class_ids,
        ) in tqdm(enumerate(test_loader), total=len(test_loader)):
            correct, total = evaluate_on_one_task(
                net, support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct

    print(
        f"Model tested on {len(test_loader)} tasks. Accuracy: {(100 * correct_predictions / total_predictions):.2f}%"
    )
    return total_predictions, correct_predictions


def train(model, train_loader, test_loader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0

        for (support_images, support_labels, query_images, query_labels, class_ids,) in train_loader:
            optimizer.zero_grad()

            outputs = model(support_images.cuda(), support_labels.cuda(), query_images.cuda()).detach()

            loss = criterion(outputs, query_labels)
            loss.backward()  # back props
            optimizer.step()  # parameter update

            # Metrics
            epoch_loss += loss
            total += len(query_labels)
            correct += (torch.max(outputs.data, 1)[1] == query_labels).sum().item()

        epoch_loss /= len(test_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(model, test_loader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for (support_images, support_labels, query_images, query_labels, class_ids,) in test_loader:
            outputs = model(support_images, support_labels, query_images)
            loss += criterion(outputs, query_labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += len(query_labels)
            correct += (predicted == query_labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total
    return loss, accuracy
