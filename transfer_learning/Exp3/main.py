import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch.nn as nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import matplotlib.pyplot as plt

# --- 新增：美化与日志库 ---
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from torch.utils.tensorboard import SummaryWriter

# --- 环境变量和随机种子 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像加速下载
torch.manual_seed(42)
np.random.seed(42)

# ==================================================================
# 0. 全局参数与工具初始化
# ==================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRE_TRAINED_MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 2e-5

# 初始化 rich Console，用于美观打印
console = Console()
console.rule("[bold green]迁移学习实验开始", style="green")
console.print(f"使用的设备: [bold cyan]{DEVICE}[/bold cyan]")

# 创建一个带时间戳的主日志目录，用于存放所有实验的 TensorBoard 日志
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_log_dir = f"runs/sentiment_transfer_{timestamp}"

# ==================================================================
# 1. 数据准备 (与原代码相同)
# ==================================================================
console.rule("[bold green]1. 数据准备", style="green")

# --- 此部分与原代码相同，保持不变 ---
import torchtext.datasets as datasets

console.print("正在加载 IMDB 数据集 (源任务)...")
try:
    train_iter, test_iter = datasets.IMDB(root=".data", split=("train", "test"))

    def get_texts_and_labels(data_iter):
        texts, labels = [], []
        for label, text in data_iter:
            texts.append(text)
            labels.append(0 if label == "neg" else 1)
        return texts, labels

    imdb_train_texts_all, imdb_train_labels_all = get_texts_and_labels(train_iter)
except Exception:
    train_iter, _ = datasets.IMDB(root=".data", split=("train", "test"))

    def get_texts_and_labels_new(data_iter):
        texts, labels = [], []
        for label, text in data_iter:
            texts.append(text)
            labels.append(label - 1)
        return texts, labels

    imdb_train_texts_all, imdb_train_labels_all = get_texts_and_labels_new(train_iter)

imdb_train_texts, _, imdb_train_labels, _ = train_test_split(
    imdb_train_texts_all,
    imdb_train_labels_all,
    train_size=5000,
    random_state=42,
    stratify=imdb_train_labels_all,
)
console.print(f"IMDB 源任务训练数据量: [yellow]{len(imdb_train_texts)}[/yellow]")


# 读取豆瓣数据
df_douban = pd.read_csv("./dataset/DOUBAN/DMSC.csv")
df_douban = df_douban.dropna(subset=["Comment", "Star"])
df_douban["label"] = df_douban["Star"].apply(lambda x: int(x) - 1)

# 先做原始训练 / 测试集划分（保持分层）
douban_train_df, douban_test_df = train_test_split(
    df_douban, test_size=0.3, random_state=42, stratify=df_douban["label"]
)

# 在训练集中采样 5000 条小样本训练集
douban_train_df, _ = train_test_split(
    douban_train_df, train_size=5000, random_state=42, stratify=douban_train_df["label"]
)

# 在测试集中采样 5000 条小样本测试集
douban_test_df, _ = train_test_split(
    douban_test_df, train_size=5000, random_state=42, stratify=douban_test_df["label"]
)

# 输出采样后的数据量
console.print(
    f"豆瓣采样后训练数据量: [yellow]{len(douban_train_df)}[/yellow], 测试数据量: [yellow]{len(douban_test_df)}[/yellow]"
)
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    ds = SentimentDataset(
        texts=texts, labels=labels, tokenizer=tokenizer, max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


imdb_train_loader = create_data_loader(
    imdb_train_texts, imdb_train_labels, tokenizer, MAX_LEN, BATCH_SIZE
)
douban_train_loader = create_data_loader(
    douban_train_df.Comment.to_numpy(),
    douban_train_df.label.to_numpy(),
    tokenizer,
    MAX_LEN,
    BATCH_SIZE,
)
douban_test_loader = create_data_loader(
    douban_test_df.Comment.to_numpy(),
    douban_test_df.label.to_numpy(),
    tokenizer,
    MAX_LEN,
    BATCH_SIZE,
)


# ==================================================================
# 2. 模型与辅助函数
# ==================================================================
console.rule("[bold green]2. 模型定义与辅助函数", style="green")


class BERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


# --- 新增：打印模型参数量的函数 ---
def print_model_parameters(model):
    table = Table(title="模型参数统计", show_header=True, header_style="bold magenta")
    table.add_column("层级", style="dim", width=30)
    table.add_column("参数量", justify="right")
    table.add_column("可训练", justify="center")

    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        is_trainable = "✅" if parameter.requires_grad else "❌"
        if parameter.requires_grad:
            trainable_params += params
        table.add_row(name, f"{params:,}", is_trainable)

    console.print(table)
    console.print(f"总参数量: [bold cyan]{total_params:,}[/bold cyan]")
    console.print(f"可训练参数量: [bold green]{trainable_params:,}[/bold green]")


# --- 重构：训练和评估函数，集成 rich.progress 和 TensorBoard ---
def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    writer,
    progress,
    task_id,
    epoch_num,
    n_examples,
):
    model = model.train()
    losses = []
    correct_predictions = 0

    for i, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 更新进度条和 TensorBoard
        global_step = epoch_num * len(data_loader) + i
        writer.add_scalar("Loss/train_step", loss.item(), global_step)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], global_step)
        progress.update(
            task_id, advance=1, description=f"Train Loss: {loss.item():.4f}"
        )

    train_acc = correct_predictions.double() / n_examples
    train_loss = np.mean(losses)
    return train_acc, train_loss


def eval_model(
    model,
    data_loader,
    loss_fn,
    device,
    writer,
    progress,
    task_id,
    epoch_num,
    n_examples,
):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            progress.update(task_id, advance=1)

    val_acc = correct_predictions.double() / n_examples
    val_loss = np.mean(losses)
    return val_acc, val_loss


# --- 新增：一个统一的训练流程函数 ---
def run_training_session(
    session_name: str,
    log_dir_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_size: int,
    val_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
):
    console.rule(f"[bold blue]开始会话: {session_name}[/bold blue]")
    print_model_parameters(model)

    # 1. 初始化 TensorBoard Writer
    writer = SummaryWriter(log_dir=os.path.join(main_log_dir, log_dir_name))

    # 将模型计算图写入 TensorBoard (取一个样本)
    try:
        sample_batch = next(iter(train_loader))
        writer.add_graph(
            model,
            (
                sample_batch["input_ids"].to(device),
                sample_batch["attention_mask"].to(device),
            ),
        )
        console.print("✅ 模型计算图已写入 TensorBoard")
    except Exception as e:
        console.print(f"❌ 无法写入模型图: {e}")

    # 2. 初始化优化器和调度器
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

    # 3. 使用 rich.progress 创建美观的进度条
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn(""),  # 分隔符
        console=console,
    ) as progress:
        epoch_task = progress.add_task("[bold cyan]总进度", total=epochs)

        for epoch in range(epochs):
            progress.update(
                epoch_task, description=f"[bold cyan]Epoch {epoch+1}/{epochs}"
            )

            # 训练
            train_task = progress.add_task(
                f"[green]  Training...", total=len(train_loader)
            )
            train_acc, train_loss = train_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                writer,
                progress,
                train_task,
                epoch,
                train_size,
            )
            progress.update(
                train_task, description=f"[green]✅ Train Acc: {train_acc:.4f}"
            )

            # 评估
            val_task = progress.add_task(
                f"[magenta]  Validating...", total=len(val_loader)
            )
            val_acc, val_loss = eval_model(
                model,
                val_loader,
                loss_fn,
                device,
                writer,
                progress,
                val_task,
                epoch,
                val_size,
            )
            progress.update(val_task, description=f"[magenta]✅ Val Acc: {val_acc:.4f}")

            # 记录历史和 TensorBoard (epoch级别)
            history["train_acc"].append(train_acc.item())
            history["train_loss"].append(train_loss.item())
            history["val_acc"].append(val_acc.item())
            history["val_loss"].append(val_loss.item())

            writer.add_scalar("Accuracy/train_epoch", train_acc, epoch)
            writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            writer.add_scalar("Accuracy/validation_epoch", val_acc, epoch)
            writer.add_scalar("Loss/validation_epoch", val_loss, epoch)

            console.print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

            progress.update(epoch_task, advance=1)

    # 4. 记录超参数和最终指标
    hparams = {
        "learning_rate": lr,
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "model": model.__class__.__name__,
    }
    final_metrics = {
        "hparam/final_val_accuracy": history["val_acc"][-1],
        "hparam/final_val_loss": history["val_loss"][-1],
    }
    writer.add_hparams(hparams, final_metrics)
    writer.close()

    #  5. 保存模型
    model_save_path = f"{log_dir_name}_model.pth"

    return history


# ==================================================================
# 3. 实验执行
# ==================================================================
console.rule("[bold green]3. 实验执行", style="green")

# --- A. 在源任务(IMDB)上微调BERT ---
IMDB_TUNED_MODEL_PATH = "imdb_finetuned_bert.bin"

if not os.path.exists(IMDB_TUNED_MODEL_PATH):
    # IMDB没有验证集，所以我们用训练集本身来评估，这里仅为演示流程
    imdb_model = BERTClassifier(n_classes=2).to(DEVICE)
    # 因为 IMDB 任务只是为了预热权重，这里就不做完整的训练会话了，保持原样快速完成
    console.rule("[bold blue]任务A: 在源任务(IMDB)上微调模型[/bold blue]")
    optimizer = AdamW(imdb_model.parameters(), lr=LEARNING_RATE)
    total_steps = len(imdb_train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    with Progress(console=console) as progress:
        epoch_task = progress.add_task("[bold cyan]IMDB 微调", total=EPOCHS)
        for epoch in range(EPOCHS):
            train_task = progress.add_task(
                f"[green]  Epoch {epoch+1}", total=len(imdb_train_loader)
            )
            # 简化版训练循环
            imdb_model.train()
            for d in imdb_train_loader:
                input_ids, attention_mask, labels = (
                    d["input_ids"].to(DEVICE),
                    d["attention_mask"].to(DEVICE),
                    d["labels"].to(DEVICE),
                )
                outputs = imdb_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(imdb_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update(
                    train_task, advance=1, description=f"Loss: {loss.item():.4f}"
                )
            progress.update(epoch_task, advance=1)

    torch.save(imdb_model.state_dict(), IMDB_TUNED_MODEL_PATH)
    console.print(
        f"✅ IMDB微调后的模型已保存至 [yellow]{IMDB_TUNED_MODEL_PATH}[/yellow]"
    )
else:
    console.print(f"✅ 已存在微调过的IMDB模型，跳过训练。")

# --- B. 方法一: 使用迁移学习在豆瓣数据集上微调 ---
transfer_model = BERTClassifier(n_classes=5).to(DEVICE)
pretrained_dict = torch.load(IMDB_TUNED_MODEL_PATH, map_location=DEVICE)
model_dict = transfer_model.state_dict()
filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("out.")}
model_dict.update(filtered_dict)
transfer_model.load_state_dict(model_dict)
console.print("✅ 迁移学习：已成功加载除分类头以外的IMDB微调权重")

history_transfer = run_training_session(
    session_name="迁移学习 (Transfer Learning)",
    log_dir_name="B_Transfer_Learning",
    model=transfer_model,
    train_loader=douban_train_loader,
    val_loader=douban_test_loader,
    train_size=len(douban_train_df),
    val_size=len(douban_test_df),
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    device=DEVICE,
)

# --- C. 方法二: 从零开始训练一个新模型 ---
scratch_model = BERTClassifier(n_classes=5).to(DEVICE)
history_scratch = run_training_session(
    session_name="从零开始训练 (From Scratch)",
    log_dir_name="C_From_Scratch",
    model=scratch_model,
    train_loader=douban_train_loader,
    val_loader=douban_test_loader,
    train_size=len(douban_train_df),
    val_size=len(douban_test_df),
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    device=DEVICE,
)

# ==================================================================
# 4. 模型评估
# ==================================================================
console.rule("[bold green]4. 模型评估", style="green")


def get_predictions(model, data_loader, device):
    model.eval()
    predictions, real_values = [], []
    with torch.no_grad():
        for d in data_loader:
            outputs = model(d["input_ids"].to(device), d["attention_mask"].to(device))
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
            real_values.extend(d["labels"])
    return torch.stack(predictions).cpu(), torch.stack(real_values).cpu()


y_pred_transfer, y_test_douban = get_predictions(
    transfer_model, douban_test_loader, DEVICE
)
y_pred_scratch, _ = get_predictions(scratch_model, douban_test_loader, DEVICE)

target_names = ["1 star", "2 star", "3 star", "4 star", "5 star"]

console.print("\n[bold] --- 迁移学习模型评估报告 --- [/bold]")
console.print(
    f"准确率 (Accuracy): {accuracy_score(y_test_douban, y_pred_transfer):.4f}"
)
console.print(
    classification_report(
        y_test_douban, y_pred_transfer, target_names=target_names, zero_division=0
    )
)

console.print("\n[bold] --- 从零训练模型评估报告 --- [/bold]")
console.print(f"准确率 (Accuracy): {accuracy_score(y_test_douban, y_pred_scratch):.4f}")
console.print(
    classification_report(
        y_test_douban, y_pred_scratch, target_names=target_names, zero_division=0
    )
)


# ==================================================================
# 5. 结果分析与可视化
# ==================================================================
console.rule("[bold green]5. 结果分析与可视化", style="green")

# 使用更美观的绘图风格
plt.style.use("seaborn-v0_8-whitegrid")

plt.figure(figsize=(14, 6))

# 使用 'viridis' colormap
cmap = plt.get_cmap("magma")
colors = cmap(np.linspace(0.1, 0.9, 4))

plt.subplot(1, 2, 1)
plt.plot(
    history_transfer["val_acc"],
    label="Transfer Learning Val Acc",
    color=colors[0],
    marker="o",
    linestyle="--",
)
plt.plot(
    history_scratch["val_acc"],
    label="From Scratch Val Acc",
    color=colors[1],
    marker="x",
)
plt.title("Validation Accuracy Comparison", fontsize=14, fontweight="bold")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.subplot(1, 2, 2)
plt.plot(
    history_transfer["val_loss"],
    label="Transfer Learning Val Loss",
    color=colors[2],
    marker="o",
    linestyle="--",
)
plt.plot(
    history_scratch["val_loss"],
    label="From Scratch Val Loss",
    color=colors[3],
    marker="x",
)
plt.title("Validation Loss Comparison", fontsize=14, fontweight="bold")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.suptitle(
    "Transfer Learning vs. From Scratch Training Performance",
    fontsize=16,
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("training_comparison_enhanced.png", dpi=300)
console.print(
    "\n✅ 训练过程对比图已保存为 [yellow]training_comparison_enhanced.png[/yellow]"
)
plt.show()

console.rule("[bold green]实验结束[/bold green]")

# --- 新增：如何查看 TensorBoard ---
console.print("\n[bold yellow]📈 如何查看 TensorBoard 日志:[/bold yellow]")
console.print("1. 在你的终端中，确保你位于此脚本所在的目录。")
console.print(
    f"2. 运行以下命令: [bold cyan]tensorboard --logdir {main_log_dir.split('/')[0]}[/bold cyan]"
)
console.print("3. 在浏览器中打开显示的网址 (通常是 http://localhost:6006/)")
