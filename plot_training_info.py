import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import os

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--model_path",  nargs='+', default="", metavar="FILE", help="path to config file")
    parser.add_argument("--log_file", default="", metavar="FILE", help="path to config file")

    args = parser.parse_args()

    return args


def loss_from_log_file(path_to_log):
    file1 = open(path_to_log, "r+")
    log_lines = file1.readlines()
    file1.close()

    recovered_train_loss_container = {}
    recovered_val_loss_container = {}
    finished_epochs = set()
    val_to_read = False
    start_epoch = False
    for line in log_lines:
        if "Starting epoch" in line and not val_to_read:
            epoch_num = int(line.split(" ")[-1].split("/")[0])
            recovered_train_loss_container[epoch_num] = []
            start_epoch = True

        elif "Finished epoch" in line:
            finished_epochs.add(epoch_num)
            start_epoch = False
            val_to_read = True

        elif start_epoch and "Loss:" in line:
            loss_value = float(line.split("\t")[-1].split(" ")[-1])
            recovered_train_loss_container[epoch_num].append(loss_value)

        elif val_to_read and "Total mean validation loss" in line:
            mean_val_loss = float(line.split(" ")[-1])
            recovered_val_loss_container[epoch_num] = mean_val_loss
            val_to_read = False

    last_read_epoch = list(recovered_train_loss_container.keys())[-1]
    if last_read_epoch not in finished_epochs:
        recovered_train_loss_container.pop(last_read_epoch)

    return recovered_train_loss_container, recovered_val_loss_container


def load_losses(path_to_pth):
    saved_training = torch.load(path_to_pth)
    return saved_training["train_loss"], saved_training["val_loss"]


def plot_validation_loss(validation_loss, out_path):
    epochs = len(validation_loss.keys())
    # mean_loss = np.array(list(validation_loss.values()))

    items_per_epoch = len(list(validation_loss.values())[0])
    mean_loss = np.array(list(itertools.chain.from_iterable(list(validation_loss.values()))))
    x_axis = np.linspace(0, epochs, num=items_per_epoch * epochs)

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.set_title("Validation Loss Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Total Loss")
    ax.plot(x_axis, mean_loss)
    out_path = os.path.join(out_path, "validation_loss.png")
    plt.savefig(out_path)


def plot_validation_loss_per_pieces(validation_loss, out_path, partitions=50):

    step = int(len(list(validation_loss.values())[0]) / partitions)
    num_epochs = len(validation_loss.keys())

    averaged_loss = []
    x_coordinates = np.linspace(0, num_epochs, num_epochs * partitions)

    for epoch in validation_loss.keys():

        loss = validation_loss[epoch]
        start_idx = 0
        for partition in range(partitions):
            averaged_loss.append(np.mean(loss[start_idx:(start_idx + step)]))
            start_idx += step

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.set_title("Smoothed Evaluation Loss Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Averaged_loss_interval")
    ax.plot(x_coordinates, np.array(averaged_loss))

    out_path = os.path.join(out_path, "smoothed_validation_loss.png")
    plt.savefig(out_path)



def plot_global_training_loss(training_loss, out_path):
    num_epochs = len(training_loss.keys())
    # individual_epoch_items
    items_per_epoch = len(list(training_loss.values())[0])

    x_axis = np.linspace(0, num_epochs, num=items_per_epoch*num_epochs)
    flattened_loss = np.array(list(itertools.chain.from_iterable(list(training_loss.values()))))

    tendency_line = np.polyfit(x_axis, flattened_loss, 2)
    p = np.poly1d(tendency_line)

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.set_title("Training Loss Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(x_axis, flattened_loss)
    ax.plot(x_axis, p(x_axis), "r--")

    out_path = os.path.join(out_path, "global_training_loss.png")
    plt.savefig(out_path)


def plot_training_loss_by_pieces(training_loss, out_path, partitions=20):
    step = int(len(list(training_loss.values())[0])/partitions)
    num_epochs = len(training_loss.keys())

    averaged_loss = []
    x_coordinates = np.linspace(0, num_epochs, num_epochs*partitions)

    for epoch in training_loss.keys():

        loss = training_loss[epoch]
        start_idx = 0
        for partition in range(partitions):
            averaged_loss.append(np.mean(loss[start_idx:(start_idx+step)]))
            start_idx += step

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.set_title("Smoothed Training Loss Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Averaged_loss_interval")
    ax.plot(x_coordinates, np.array(averaged_loss))

    out_path = os.path.join(out_path, "smoothed_training_loss.png")
    plt.savefig(out_path)



def plot_training_loss_per_epoch(training_loss, out_path):
    fig, ax = plt.subplots()
    ax.set_title("TRAINING LOSS PER EPOCH")
    ax.set_xlabel("Iter_in_Epoch")
    ax.set_ylabel("Loss")

    for epoch in training_loss.keys():
        loss = training_loss[epoch]
        num_items = np.arange(len(loss))

        ax.plot(num_items, np.array(loss), label='EPOCH: {}'.format(epoch))

    ax.legend()

    out_path = os.path.join(out_path, "per_epoch_training_loss.png")
    plt.savefig(out_path)

def get_smoothed_training_loss(training_loss, partitions=20):

    step = int(len(list(training_loss.values())[0]) / partitions)
    averaged_loss = []

    for epoch in training_loss.keys():

        loss = training_loss[epoch]
        start_idx = 0
        for partition in range(partitions):
            averaged_loss.append(np.mean(loss[start_idx:(start_idx + step)]))
            start_idx += step

    return averaged_loss

def get_average_val_loss(validation_loss, partitions):
    total_mean_loss = []
    for epoch in validation_loss.keys():
        mean_loss = np.array(np.mean(validation_loss[epoch]))
        for _ in range(partitions):
            total_mean_loss.append(mean_loss)

    return total_mean_loss


def prepare_losses_dict(models_to_read):
    losses_dict = {}
    num_epochs = None
    for model in models_to_read:
        name = model.split("/")[-1].split(".")[0]

        train_loss, val_loss = load_losses(model)
        if num_epochs is None:
            num_epochs = len(train_loss.keys())

        losses_dict[name] = {}
        losses_dict[name]["training_loss"] = train_loss
        losses_dict[name]["validation_loss"] = val_loss

    return losses_dict, num_epochs


def get_train_val_values(losses, partitions, num_epochs, out_path):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.set_title("Losses plot")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    x_coordinates = np.linspace(0, num_epochs, num_epochs * partitions)

    for model, losses in losses.items():
        training_loss = losses["training_loss"]
        validation_loss = losses["validation_loss"]

        smoothed_training_loss = get_smoothed_training_loss(training_loss, partitions)
        mean_val_loss = get_average_val_loss(validation_loss, partitions)

        ax.plot(x_coordinates, np.array(smoothed_training_loss), label='Training_loss: {}'.format(model))
        ax.plot(x_coordinates, np.array(mean_val_loss), label='Validation_loss: {}'.format(model))

    ax.legend()

    out_path = os.path.join(out_path, "all_train_val_loss.png")
    plt.savefig(out_path)



if __name__ == "__main__":
    args = argument_parser()
    # if args.model_path:
    #     print("Reading loss from passed checkpoint")
    #     loaded_training_loss, loaded_validation_loss = load_losses(args.model_path)
    # elif args.log_file:
    #     print("Reading loss from log-file")
    #     loaded_training_loss, loaded_validation_loss = loss_from_log_file(args.log_file)
    #
    # else:
    #     raise ValueError("No path to read loss from specified!")

    # plot_training_loss_per_epoch(training_loss, args.out_path)
    # plot_global_training_loss(loaded_training_loss, args.out_path)
    # plot_training_loss_by_pieces(loaded_training_loss, args.out_path)
    # plot_validation_loss(loaded_validation_loss, args.out_path)
    # plot_validation_loss_per_pieces(loaded_validation_loss, args.out_path)

    losses_dict, num_epochs = prepare_losses_dict(args.model_path)
    get_train_val_values(losses_dict, 20, num_epochs, args.out_path)