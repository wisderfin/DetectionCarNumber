import torch

# image
SIZE_IMAGE = (32, 128)
CHANELS_IMAGE = 1

# symbols
CHARS = 'ABEKMHOPTHCNYX0123456789'
CHARS_SIZE = len(CHARS) + 1
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}

# train
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
