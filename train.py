import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse

from SpiderUtils.dataset import SpiderDataset
from SpiderUtils.models import Spider
from SpiderUtils.Cosine_warm import CosineWarmupDecay
plt.switch_backend('agg')

Loss = nn.CrossEntropyLoss(reduction='mean')

def loss_fn(preds, labels, mask):
    pred = preds[mask]
    label = labels[mask]
    loss = Loss(pred, label)
    return loss
    

def accuracy_fn(preds, labels, mask):
    preds = preds[mask]
    labels = labels[mask]
    correct_prediction = torch.argmax(preds, dim=1) == torch.argmax(labels, dim=1)
    accuracy_all = correct_prediction.float()
    accuracy = accuracy_all.sum() / len(accuracy_all)
    return accuracy.item()


def train_one_epoch(model, data, optimizer):
    model.train()
    out = model(data)
    loss = loss_fn(out, data.y, data.train_mask)
    acc = accuracy_fn(out.cpu().detach(), data.y.cpu(), data.train_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc


@torch.no_grad()
def eval_one_epoch(model, data):
    model.eval()
    out = model(data)
    mask = data.test_mask
    loss = loss_fn(out, data.y, mask)
    acc = accuracy_fn(out.cpu().detach(), data.y.cpu(), mask)
    return loss, acc


@torch.no_grad()
def predict(model, data):
    model.eval()
    out = model(data)
    return F.softmax(out, dim=-1)


def draw_history(history: dict):
    plt.figure()
    plt.subplot(121)
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(history["acc"], label="acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "history.png"))
    plt.close()


def train_loop(epochs, patience, model, data, optimizer, scheduler=None, draw_after=True):
    history = {
        "loss": [],
        "acc": [],
        "val_loss": [],
        "val_acc": []
    }

    early_stop_step = 0
    val_acc_max = 0
    val_loss_min = float('inf')
    best_epoch = 0

    for epoch in tqdm(range(epochs)):
        loss, acc = train_one_epoch(model, data, optimizer)
        val_loss, val_acc = eval_one_epoch(model, data)
        if scheduler:
            scheduler.step()

        history["loss"].append(loss.item())
        history["acc"].append(acc)
        history["val_loss"].append(val_loss.item())
        history["val_acc"].append(val_acc)

        if not draw_after:
            draw_history(history)

        if val_acc >= val_acc_max or val_loss <= val_loss_min:
            if val_acc > val_acc_max and val_loss < val_loss_min:
                val_acc_model = val_acc
                val_loss_model = val_loss
                trian_acc_model = acc
                train_loss_model = loss
                best_epoch = epoch
                torch.save(model.state_dict(), BEST_MODEL_PATH)

            val_acc_max = max(val_acc, val_acc_max)
            val_loss_min = min(val_loss, val_loss_min)
            early_stop_step = 0
        else:
            early_stop_step += 1
            if early_stop_step >= patience:
                print("Early stopping.")
                print(f"Best val acc@ {best_epoch}: {val_acc_model} training acc @ {best_epoch} : {trian_acc_model}")
                print(f"Best val loss: {val_loss_model} training loss: {train_loss_model}")
                break
    
    draw_history(history)
    torch.save(model.state_dict(), LAST_MODEL_PATH)
    print(f"Best val acc@ {best_epoch}: {val_acc_model} training acc @ {best_epoch} : {trian_acc_model}")
    print(f"Best val loss: {val_loss_model} training loss: {train_loss_model}")


def result_infer(model, data, data_dir):
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    res = predict(model, data)
    res = res.cpu().numpy()[data.pred_mask]
    column_names = pd.read_csv(os.path.join(data_dir, "ST_label/ST_label_1.csv"), index_col=0).columns
    res = pd.DataFrame(res, columns=column_names, index=None)
    res.to_csv(BEST_RESULT_PATH, index=None)

    model.load_state_dict(torch.load(LAST_MODEL_PATH))
    res_last = predict(model, data)
    res_last = res_last.cpu().numpy()[data.pred_mask]
    res_last = pd.DataFrame(res_last, columns=column_names, index=None)
    res_last.to_csv(LAST_RESULT_PATH, index=None)

    print(f"Inference finished. Results saved to {BEST_RESULT_PATH} and {LAST_RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/PDAC-A")
    args = parser.parse_args()

    data_type = args.data_root
    
    data_root = f"{data_type}/torch_data"
    raw_data_root = f"{data_type}/Infor_Data"
    
    save_dir = f"{data_type}/results"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    BEST_MODEL_PATH = os.path.join(save_dir, "best_model.pth")
    LAST_MODEL_PATH = os.path.join(save_dir, "last.pth")
    BEST_RESULT_PATH = os.path.join(save_dir, "best_result.csv")
    LAST_RESULT_PATH = os.path.join(save_dir, "result_last.csv")

    Epochs = 1000
    Patience = 300
    Learning_Rate = 0.01
    Weight_Decay = 0.0
    hid_units = [128, 128]
    n_heads = [8, 8]
    dropout=0.1
    draw_after = True
    force_reload = True


    dataset = SpiderDataset(root=data_root, raw_data_root=raw_data_root, force_reload=force_reload)
    model = Spider(
        input_num_features = dataset.num_features, 
        hid_units = hid_units, 
        n_heads = n_heads, 
        nb_classes=dataset.num_classes,
        edge_dim=1,
        dropout=dropout,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)
    scheduler = CosineWarmupDecay(optimizer, warmup_step=5, total_step=20, multi=0.25, print_step=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = dataset[0].to(device)
    train_loop(Epochs, Patience, model, data, optimizer, scheduler, draw_after=draw_after)
    result_infer(model, data, raw_data_root)

