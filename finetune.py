import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"

CHECKPOINT = "sam_vit_b.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 1024
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4


# ----------------------------------------------------
# Dice Loss
# ----------------------------------------------------

class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)

        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()

        dice = (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice


# ----------------------------------------------------
# Dataset
# ----------------------------------------------------

class CropDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def get_box(self, mask):

        y, x = np.where(mask > 0)

        xmin = np.min(x)
        xmax = np.max(x)

        ymin = np.min(y)
        ymax = np.max(y)

        return np.array([xmin, ymin, xmax, ymax])

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image = cv2.imread(
            os.path.join(self.image_dir, image_name)
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        mask = cv2.imread(
            os.path.join(MASK_DIR, image_name),
            0
        )

        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

        mask = (mask > 127).astype(np.float32)

        box = self.get_box(mask)

        image = torch.tensor(image).permute(2,0,1).float()/255

        mask = torch.tensor(mask).unsqueeze(0)

        box = torch.tensor(box).float()

        return image, mask, box


# ----------------------------------------------------
# DataLoader
# ----------------------------------------------------

dataset = CropDataset(
    IMAGE_DIR,
    MASK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ----------------------------------------------------
# Load SAM
# ----------------------------------------------------

sam = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT
)

sam.to(DEVICE)

# ----------------------------------------------------
# Freeze Image Encoder
# ----------------------------------------------------

for param in sam.image_encoder.parameters():
    param.requires_grad = False

# ----------------------------------------------------
# Optimizer
# ----------------------------------------------------

optimizer = torch.optim.Adam(

    list(sam.prompt_encoder.parameters()) +

    list(sam.mask_decoder.parameters()),

    lr=LR
)

dice_loss = DiceLoss()

bce_loss = nn.BCEWithLogitsLoss()

# ----------------------------------------------------
# Training
# ----------------------------------------------------

sam.train()

for epoch in range(EPOCHS):

    epoch_loss = 0

    for images, gt_masks, boxes in loader:

        images = images.to(DEVICE)
        gt_masks = gt_masks.to(DEVICE)
        boxes = boxes.to(DEVICE)

        optimizer.zero_grad()

        image_embeddings = sam.image_encoder(images)

        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None
        )

        pred_masks, iou_predictions = sam.mask_decoder(

            image_embeddings=image_embeddings,

            image_pe=sam.prompt_encoder.get_dense_pe(),

            sparse_prompt_embeddings=sparse_embeddings,

            dense_prompt_embeddings=dense_embeddings,

            multimask_output=False
        )

        loss = (
            dice_loss(pred_masks, gt_masks)
            +
            bce_loss(pred_masks, gt_masks)
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch+1} Loss : {epoch_loss/len(loader):.4f}"
    )

torch.save(
    sam.state_dict(),
    "sam_crop_finetuned.pth"
)

print("Training Complete")
