"""
This is a minimal example script for Visual Question Answering using the BigEarthNet dataset.
It demonstrates how to train a model on the RSVQAxBEN dataset using the ConfigILM library.
The relevant parts of the codes are
 - lines 30-38 (model setup) and
 - lines 40-63 (dataset setup).
The rest of the code is boilerplate code for training and evaluation.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra import BEN_lmdb_utils
from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform
from configilm.extra.DataSets.RSVQAxBEN_DataSet import resolve_data_dir
from configilm.extra.DataSets.RSVQAxBEN_DataSet import RSVQAxBENDataSet

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"

image_model = "resnet18"
text_model = "prajjwal1/bert-tiny"
number_of_channels = 3
image_size = 120
num_workers = 4

model_config = ILMConfiguration(
    timm_model_name=image_model,
    hf_model_name=text_model,
    classes=1000,
    image_size=image_size,
    channels=number_of_channels,
    network_type=ILMType.VQA_CLASSIFICATION,
)
model = ConfigILM.ConfigILM(model_config)

ben_mean, ben_std = BEN_lmdb_utils.band_combi_to_mean_std(number_of_channels)
train_transform = default_train_transform(img_size=(image_size, image_size), mean=ben_mean, std=ben_std)
transform = default_transform(img_size=(image_size, image_size), mean=ben_mean, std=ben_std)

train_ds = RSVQAxBENDataSet(
    resolve_data_dir(None, allow_mock=True),
    split="train",
    transform=train_transform,
    img_size=(number_of_channels, image_size, image_size),
)

val_ds = RSVQAxBENDataSet(
    resolve_data_dir(None, allow_mock=True),
    split="val",
    transform=transform,
    img_size=(number_of_channels, image_size, image_size),
)

test_ds = RSVQAxBENDataSet(
    resolve_data_dir(None, allow_mock=True),
    split="test",
    transform=transform,
    img_size=(number_of_channels, image_size, image_size),
)

train_dl = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
)
val_dl = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
)
test_dl = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
)

optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
loss_fn = F.binary_cross_entropy_with_logits

for epoch in range(10):
    running_loss = 0.0
    # train loop
    model.train()
    for i, batch in enumerate(train_dl):
        img, text, label = batch

        optim.zero_grad()
        output = model((img, torch.stack(text).T))
        loss = loss_fn(output, label)
        loss.backward()
        optim.step()
        if i % 10 == 9:  # print every 10 mini-batches
            print(f"[{epoch + 1:3d}, {i + 1:5d}] loss: {running_loss / 10:.4f}")
            running_loss = 0.0

    # validation
    model.eval()
    eval_loss = 0.0
    for i, batch in enumerate(val_dl):
        img, text, label = batch
        output = model((img, torch.stack(text).T))
        loss = loss_fn(output, label)
        eval_loss += loss.item()
    eval_loss /= len(val_dl)
    print(f"Epoch {epoch + 1:3d} validation loss: {eval_loss:.4f}")

# test
model.eval()
test_loss = 0.0
for i, batch in enumerate(test_dl):
    img, text, label = batch
    output = model((img, torch.stack(text).T))
    loss = loss_fn(output, label)
    test_loss += loss.item()
test_loss /= len(test_dl)
print(f"Test loss: {test_loss:.4f}")
