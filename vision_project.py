import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, LongformerModel, LongformerConfig

# Data loading and preprocessing
dataset_slug = "flickr30k_images"
file_path = f"/kaggle/input/{dataset_slug}/flickr30k_images/results.csv"
if os.path.exists(file_path):
    print(f"Found file at {file_path}")
else:
    print(f"File not found at {file_path}")
    print("Available datasets:", os.listdir("/kaggle/input/"))
    print("Contents of dataset:", os.listdir(f"/kaggle/input/{dataset_slug}/"))
    raise FileNotFoundError("Check the dataset slug and file path!")
df = pd.read_csv(file_path, delimiter="|")
print("First few rows:")
print(df.head())
df.columns = ['image', 'caption_number', 'caption']
df['caption'] = df['caption'].str.lstrip()
df['caption_number'] = df['caption_number'].str.lstrip()
if len(df) > 19999:
    df.loc[19999, 'caption_number'] = "4"
    df.loc[19999, 'caption'] = "A dog runs across the grass ."
else:
    print(f"Warning: DataFrame has only {len(df)} rows, cannot modify index 19999.")
ids = [id_ for id_ in range((len(df) + 4) // 5) for i in range(5)][:len(df)]
df['id'] = ids
output_path = "/kaggle/working/captions.csv"
df.to_csv(output_path, index=False)
print(f"File saved to {output_path}")
df = pd.read_csv("/kaggle/input/flickr30k_images/flickr30k_images/results.csv", delimiter="|")
df.columns = ['image', 'caption_number', 'caption']
df['caption'] = df['caption'].str.lstrip()
df['caption_number'] = df['caption_number'].str.lstrip()
df.loc[19999, 'caption_number'] = "4"
df.loc[19999, 'caption'] = "A dog runs across the grass ."
ids = [id_ for id_ in range(len(df) // 5) for i in range(5)]
df['id'] = ids
df.to_csv("captions.csv", index=False)

# CFG class with adjustments
class CFG:
    debug = False
    image_path = "/kaggle/input/flickr30k_images/flickr30k_images"
    captions_path = "."
    batch_size = 8  # Reduced from 32 to 8 for stability
    num_workers = 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2  # Kept at 2 for accurate performance evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 64
    pretrained = True
    trainable = True
    temperature = 1.0
    size = 224
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

cfg = CFG()

# AvgMeter class
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg = self.sum = self.count = 0

    def update(self, val, count=1):
        if not isinstance(val, (int, float)):
            raise ValueError(f"Expected numeric value for {self.name}, got {type(val)}")
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# CLIPDataset class
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}
        image_path = os.path.join(CFG.image_path, self.image_filenames[idx])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be loaded: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["caption"] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)

class ImageEncoder(nn.Module):
    def __init__(self, model_name=cfg.model_name, pretrained=cfg.pretrained, trainable=cfg.trainable):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class CustomTextEncoder(nn.Module):
    def __init__(self, attention_variant: str = "self", model_name: str = cfg.text_encoder_model, 
                 pretrained: bool = cfg.pretrained, trainable: bool = cfg.trainable):
        super().__init__()
        self.attention_variant = attention_variant.lower()
        if self.attention_variant == "self":
            self.model = DistilBertModel.from_pretrained(model_name) if pretrained else DistilBertModel(config=DistilBertConfig())
        elif self.attention_variant == "cross":
            self.model = DistilBertModel.from_pretrained(model_name) if pretrained else DistilBertModel(config=DistilBertConfig())
            self.image_projection = nn.Linear(cfg.image_embedding, cfg.text_embedding)
        elif self.attention_variant == "sparse":
            sparse_model_name = "allenai/longformer-base-4096"
            self.model = LongformerModel.from_pretrained(sparse_model_name) if pretrained else LongformerModel(config=LongformerConfig())
        else:
            raise ValueError(f"Unsupported attention variant: {self.attention_variant}")
        for p in self.model.parameters():
            p.requires_grad = trainable
        if self.attention_variant == "cross":
            self.image_projection.requires_grad_(trainable)
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask, image_features=None):
        if self.attention_variant == "self":
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return output.last_hidden_state[:, self.target_token_idx, :]
        elif self.attention_variant == "cross":
            if image_features is None:
                raise ValueError("image_features required for cross-attention")
            image_token = self.image_projection(image_features).unsqueeze(1)
            embeddings = self.model.embeddings(input_ids)
            embeddings = torch.cat([embeddings, image_token], dim=1)
            extended_attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0), 1).to(attention_mask.device)], dim=1)
            output = self.model(inputs_embeds=embeddings, attention_mask=extended_attention_mask)
            return output.last_hidden_state[:, self.target_token_idx, :]
        elif self.attention_variant == "sparse":
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
            return output.last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=cfg.projection_dim, dropout=cfg.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(self, attention_variant: str = "self", temperature=cfg.temperature,
                 image_embedding=cfg.image_embedding, text_embedding=cfg.text_embedding):
        super().__init__()
        self.attention_variant = attention_variant
        self.image_encoder = ImageEncoder()
        self.text_encoder = CustomTextEncoder(attention_variant=attention_variant)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        if self.attention_variant == "cross":
            text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], image_features=image_features)
        else:
            text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss  # Return unreduced loss tensor

# Data loading and training utilities
def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{cfg.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not cfg.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(image_ids, size=int(0.2 * len(image_ids)), replace=False)
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(dataframe["image"].values, dataframe["caption"].values, tokenizer=tokenizer, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=(mode == "train"))
    return dataloader

def get_transforms(mode="train"):
    return A.Compose([A.Resize(cfg.size, cfg.size, always_apply=True), A.Normalize(max_pixel_value=255.0, always_apply=True)])

# Training and validation functions
from torch import amp

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter("Train Loss")
    tqdm_object = tqdm(train_loader, total=len(train_loader), desc="Training Epoch", unit="batch")
    scaler = amp.GradScaler('cuda')
    for i, batch in enumerate(tqdm_object):
        print(f"Processing batch {i+1}/{len(train_loader)}")
        batch = {k: v.to(cfg.device) for k, v in batch.items() if k != "caption"}
        optimizer.zero_grad()
        with amp.autocast('cuda'):
            loss = model(batch)
            loss = loss.mean()  # Reduce loss explicitly here
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == "batch":
            lr_scheduler.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter("Valid Loss")
    accuracy_meter = AvgMeter("Accuracy")
    recall_at_5_meter = AvgMeter("Recall@5")
    inference_time_meter = AvgMeter("Inference Time")
    memory_usage_meter = AvgMeter("Memory Usage")
    model.eval()
    with torch.no_grad():
        tqdm_object = tqdm(valid_loader, total=len(valid_loader), desc="Validation Epoch", unit="batch")
        for batch in tqdm_object:
            start_time = time.time()
            batch = {k: v.to(cfg.device) for k, v in batch.items() if k != "caption"}
            loss = model(batch)
            loss = loss.mean()  # Reduce loss explicitly here
            end_time = time.time()
            inference_time = end_time - start_time
            image_features = model.image_encoder(batch["image"])
            text_features = model.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                image_features=image_features if model.attention_variant == "cross" else None
            )
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)
            logits = (text_embeddings @ image_embeddings.T) / model.temperature
            preds = torch.argmax(logits, dim=1)
            targets = torch.arange(len(batch["image"])).to(cfg.device)
            accuracy = (preds == targets).float().mean().item()
            top5_preds = torch.topk(logits, k=5, dim=1).indices
            recall_at_5 = (top5_preds == targets.unsqueeze(1)).any(dim=1).float().mean().item()
            memory_usage = torch.cuda.memory_allocated() / 1024**2
            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            accuracy_meter.update(accuracy, count)
            recall_at_5_meter.update(recall_at_5, count)
            inference_time_meter.update(inference_time, count)
            memory_usage_meter.update(memory_usage, count)
            tqdm_object.set_postfix(valid_loss=loss_meter.avg, accuracy=accuracy_meter.avg)
    return {
        "loss": loss_meter.avg,
        "accuracy": accuracy_meter.avg,
        "recall_at_5": recall_at_5_meter.avg,
        "avg_inference_time": inference_time_meter.avg,
        "avg_memory_usage": memory_usage_meter.avg
    }

# Training loop
import time
results = {}
attention_variants = ["sparse"]
train_df, valid_df = make_train_valid_dfs()
tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
train_loader = build_loaders(train_df, tokenizer, mode="train")
valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

for variant in attention_variants:
    print(f"\nTraining model with {variant} attention")
    model = CLIPModel(attention_variant=variant)
    # Temporarily disable multi-GPU for stability
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)
    model = model.to(cfg.device)
    image_encoder_params = model.image_encoder.parameters()
    text_encoder_params = model.text_encoder.parameters()
    projection_params = itertools.chain(model.image_projection.parameters(), model.text_projection.parameters())
    params = [
        {"params": image_encoder_params, "lr": cfg.image_encoder_lr},
        {"params": text_encoder_params, "lr": cfg.text_encoder_lr},
        {"params": projection_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=cfg.patience, factor=cfg.factor)
    step = "epoch"
    best_loss = float('inf')
    train_losses = []
    valid_metrics = []
    epoch_times = []
    for epoch in range(cfg.epochs):
        print(f"\nEpoch: {epoch + 1}/{cfg.epochs}")
        start_time = time.time()
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        train_losses.append(train_loss.avg)
        print(f"Epoch {epoch + 1} training completed in {epoch_time:.2f} seconds")
        model.eval()
        with torch.no_grad():
            valid_metric = valid_epoch(model, valid_loader)
            valid_metrics.append(valid_metric)
        if valid_metric["loss"] < best_loss:
            best_loss = valid_metric["loss"]
            torch.save(model.state_dict(), f"best_{variant}.pt")
            print(f"Saved Best Model for {variant} attention with loss: {best_loss:.4f}")
        lr_scheduler.step(valid_metric["loss"])
    results[variant] = {"train_losses": train_losses, "valid_metrics": valid_metrics, "epoch_times": epoch_times}
    del model, optimizer, lr_scheduler
    torch.cuda.empty_cache()

# Print Results
for variant in attention_variants:
    print(f"\n{variant.capitalize()} Attention Metrics:")
    print(f"Average Training Loss: {np.mean(results[variant]['train_losses']):.4f}")
    print(f"Average Validation Loss: {np.mean([m['loss'] for m in results[variant]['valid_metrics']]):.4f}")
    print(f"Average Validation Accuracy: {np.mean([m['accuracy'] for m in results[variant]['valid_metrics']]):.4f}")
    print(f"Average Validation Recall@5: {np.mean([m['recall_at_5'] for m in results[variant]['valid_metrics']]):.4f}")
    print(f"Average Inference Time per Batch: {np.mean([m['avg_inference_time'] for m in results[variant]['valid_metrics']]):.4f} seconds")
    print(f"Average Memory Usage: {np.mean([m['avg_memory_usage'] for m in results[variant]['valid_metrics']]):.2f} MB")
    print(f"Average Training Time per Epoch: {np.mean(results[variant]['epoch_times']):.2f} seconds")