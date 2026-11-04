import cupy as cp
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from cuml.manifold import UMAP
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassRecall,
)

import wandb
from base import BaseTrainer
from utils import MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader,
        lr_scheduler,
        do_validation=True,
    ):
        super().__init__(
            model,
            criterion,
            metric_ftns,
            optimizer,
            config,
            device,
            data_loader,
            valid_data_loader,
            lr_scheduler,
        )
        self.len_epoch = config["trainer"].get("len_epoch", 5000)
        self.do_validation = do_validation

        self.train_data_iter = iter(self.data_loader)

        self.scaler = torch.amp.GradScaler("cuda")

        self.global_train_step = 0
        self.global_val_step = 0
        self.early_steps = config["trainer"].get("early_steps", 8)
        self.early_weight = config["trainer"].get("early_weight", 0.1)
        self.start_epoch = 1

        wandb_id = None
        if self.resume_ckpt is not None:
            self.start_epoch = self.resume_ckpt.get("epoch", 1)
            self.global_train_step = self.resume_ckpt.get("global_train_step", 0)
            self.global_val_step = self.resume_ckpt.get("global_val_step", 0)
            wandb_id = self.resume_ckpt.get("wandb_id")

        self.train_metrics = MetricTracker(
            "loss",
            "accuracy",
            "plain",
            "logit",
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
            section="train",
            use_wandb=config["use_wandb"],
        )
        self.dbg_metrics = MetricTracker(
            "dbg/unique_classes",
            "dbg/top_class_share",
            writer=self.writer,
            section="dbg",
            use_wandb=config["use_wandb"],
        )
        self.valid_metrics = MetricTracker(
            "loss",
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
            section="val",
            use_wandb=config["use_wandb"],
        )

        if config["use_wandb"]:
            # NOTE: no worry reinit
            #             wandb: WARNING Ignoring project 'limlim' when running a sweep.
            # wandb: wandb.init() called while a run is active and reinit is set to 'default', so returning the previous run.

            wandb_kwargs = {
                "project": config["name"],
                "config": config.config,
            }
            if self.resume_ckpt is not None and wandb_id:
                wandb_kwargs["id"] = wandb_id
                wandb_kwargs["resume"] = "allow"

            wandb.init(**wandb_kwargs)

            wandb.define_metric("global_train_step")
            wandb.define_metric("train_*", step_metric="global_train_step")
            wandb.define_metric("val/*", step_metric="global_train_step")
            wandb.define_metric("dbg/*", step_metric="global_train_step")

            wandb.define_metric("epoch")
            wandb.define_metric("ep/*", step_metric="epoch")

        # init metric param
        self.C = self.data_loader.num_classes
        self.f1_flow_metric = MulticlassF1Score(num_classes=self.C, average="macro").to(
            self.device
        )
        self.cm_metric = MulticlassConfusionMatrix(
            num_classes=self.C, normalize=None
        ).to(device)
        self.f1_step_metric = MulticlassF1Score(num_classes=self.C, average="macro").to(
            self.device
        )
        self.recall_flow_metric = MulticlassRecall(
            num_classes=self.C, average="macro"
        ).to(self.device)
        self.n_bins = 15  # ECE bins
        self._reset_val_stats()

    def _reset_val_stats(self):
        # num pred
        self.ece_n = torch.zeros(self.n_bins, dtype=torch.long, device=self.device)
        # sum pred
        self.ece_conf = torch.zeros(
            self.n_bins, dtype=torch.float64, device=self.device
        )
        # num true pred
        self.ece_acc = torch.zeros(self.n_bins, dtype=torch.float64, device=self.device)

        self.step_hits = torch.zeros(1024, device=self.device)
        self.step_total = torch.zeros(1024, device=self.device)
        self.f1_flow_metric.reset()
        self.f1_step_metric.reset()
        self.recall_flow_metric.reset()
        self.cm_metric.reset()

    def _accumulate(self, plain, target, mask):
        """
        B*L, C
        B
        B, L
        """
        B, L = mask.shape
        conf, pred = plain.softmax(1).max(1)  # B*L
        conf, pred = conf.view(B, L), pred.view(B, L)
        correct = pred == target.unsqueeze(1)

        # confusion matrices
        t_flat = target.unsqueeze(1).expand(B, L)[mask]
        p_flat = pred[mask]
        self.cm_metric.update(p_flat, t_flat)
        self.f1_step_metric.update(p_flat, t_flat)

        # f1 last pkgs
        lens = mask.sum(1)  # B, 1
        ok = lens > 0
        last = (lens - 1).clamp(min=0)
        ar = torch.arange(B, device=plain.device)
        flow_pred = pred[ar, last][ok]
        flow_target = target[ok]
        self.f1_flow_metric.update(flow_pred, flow_target)
        self.recall_flow_metric.update(flow_pred, flow_target)

        c_m = conf[mask]
        bins = (c_m * self.n_bins).long().clamp(max=self.n_bins - 1)
        self.ece_n += torch.bincount(bins, minlength=self.n_bins)
        self.ece_conf += torch.bincount(
            bins, weights=c_m, minlength=self.n_bins
        ).double()
        self.ece_acc += torch.bincount(
            bins, weights=correct[mask].float(), minlength=self.n_bins
        ).double()

    @staticmethod
    def _confusion_matrix(cm):
        cm_np = cm.detach().cpu().numpy()
        cm_norm = cm_np / cm_np.sum(axis=1, keepdims=True).clip(min=1e-8)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_norm,
            annot=False,
            cmap="Blues",
            cbar=True,
            ax=ax,
        )
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title("Normalized Flow Confusion Matrix")
        plt.tight_layout()
        plt.close(fig)
        return wandb.Image(fig)

    def _val_summary(self, acc_t, alive, epoch):
        f1_flow = self.f1_flow_metric.compute().item()  # Remember to call .item()
        bal_flow = self.recall_flow_metric.compute().item()
        f1_step = self.f1_step_metric.compute().item()
        cm_tensor = self.cm_metric.compute()  # Raw tensor

        n = self.ece_n.double().clamp(min=1)
        N = self.ece_n.sum().item()
        ece = (
            (self.ece_acc / n - self.ece_conf / n).abs() * self.ece_n
        ).sum().item() / max(N, 1)

        out = {
            "macro_f1_flow": f1_flow,
            "bal_acc_flow": bal_flow,
            "macro_f1_step": f1_step,
            "ece": ece,
            "cm_plot": self._confusion_matrix(cm_tensor)
            if self.config["use_wandb"]
            else None,
        }
        return out

    def _forward(self, data, target, mask, train):
        B, C, L = data.shape
        flat_target = target.repeat_interleave(L)
        flat_mask = mask.reshape(-1)

        # B*L  | B*L,N
        logits, plain, emb = self.model(data, target if train else None)

        # B*L
        loss = self.criterion(logits, flat_target, reduction="none")
        # L
        mults = (
            torch.clamp(torch.arange(L, device=self.device) + 1, max=self.early_steps)
            * self.early_weight
        )
        weights = mults.repeat(B) * flat_mask.float()
        loss = (loss * weights).sum() / flat_mask.sum().clamp(min=1)

        if not train:
            hit = (plain.argmax(1).view(B, L) == target.unsqueeze(1)) & mask
            self.step_hits[:L] += hit.sum(0).float()
            self.step_total[:L] += mask.sum(0).float()
            self._accumulate(plain, target, mask)

        return (
            loss,
            plain[flat_mask],
            flat_target[flat_mask],
            emb,
            logits,
            plain,
        )

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        gn_sum = 0.0
        warmup_epochs = self.config["trainer"]["warmup_epochs"]
        ramp_length = 5
        ramp_end = warmup_epochs + ramp_length  # 6

        if epoch <= warmup_epochs:
            current_m = 0.0
        elif epoch <= ramp_end:
            progress = (epoch - warmup_epochs) / ramp_length
            current_m = self.config["arch"]["m"] * progress
        else:
            current_m = self.config["arch"]["m"]
        if epoch < warmup_epochs:
            progress = (epoch + 1) / warmup_epochs
            self.optimizer.param_groups[0]["lr"] = (
                self.config["optimizer"]["lr"] * progress
            )
        self.model.m.fill_(current_m)
        for batch_idx in range(self.len_epoch):
            try:
                data, target, mask = next(self.train_data_iter)
            except StopIteration:
                self.train_data_iter = iter(self.data_loader)
                data, target, mask = next(self.train_data_iter)
            data, target, mask = (
                data.to(self.device, non_blocking=True),
                target.to(self.device, non_blocking=True),
                mask.to(self.device, non_blocking=True),
            )

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, output, flat_target, _, logit, plain = self._forward(
                    data, target, mask, train=True
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            gn_sum += float(gn)
            self.scaler.step(self.optimizer)

            self._train_epoch_metric_update(
                self.global_train_step, output, flat_target, loss, plain, logit
            )

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    f"Train Epoch: {epoch} [{batch_idx}/{self.len_epoch}] Loss: {loss.item():.6f}"
                )
            self.global_train_step += 1

        log = self.train_metrics.result()
        log["grad_norm"] = gn_sum / max(self.len_epoch, 1)
        log["lr"] = self.optimizer.param_groups[0]["lr"]

        if self.do_validation and epoch % 5 == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.config["use_wandb"]:
            safe_log = {"epoch": epoch}
            for k, v in log.items():
                if hasattr(v, "mean") and not np.isscalar(v):
                    safe_log[f"ep/{k}"] = float(v.mean())
                elif hasattr(v, "item"):
                    safe_log[f"ep/{k}"] = v.item()
                else:
                    safe_log[f"ep/{k}"] = float(v)
            wandb.log(safe_log)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self._reset_val_stats()  # NEW

        run_tsne = self.config["use_wandb"]
        all_embeddings = []
        all_targets = []
        tsne_per_batch = 300

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch_idx, (data, target, mask) in enumerate(self.valid_data_loader):
                data, target, mask = (
                    data.to(self.device),
                    target.to(self.device),
                    mask.to(self.device),
                )

                loss, output, flat_target, emb, _, _ = self._forward(
                    data, target, mask, train=False
                )

                if run_tsne:
                    B, C, L = data.shape
                    D = emb.shape[-1]
                    target = (
                        target.unsqueeze(1).expand(B, L)
                        if target.dim() == 1
                        else target
                    )

                    emb_trimmed = emb[:, -5:, :]  # [B, 20, dim]
                    target_trimmed = target[:, -5:]  # [B, 20]
                    mask_trimmed = mask[:, -5:]  # [B, 20]

                    flat_mask = mask_trimmed.reshape(-1)
                    e = emb_trimmed.reshape(-1, D)[flat_mask]
                    t = target_trimmed.reshape(-1)[flat_mask]

                    if e.shape[0] > 0:
                        if e.shape[0] > tsne_per_batch:
                            idx = torch.randperm(e.shape[0], device=e.device)[
                                :tsne_per_batch
                            ]
                            e, t = e[idx], t[idx]

                        all_embeddings.append(e.float())
                        all_targets.append(t)

                self.writer.set_step(self.global_val_step, "valid")
                self.global_val_step += 1
                self.valid_metrics.update("loss", loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, flat_target))
                del data, target, mask, loss, output, flat_target
                if batch_idx == 512:
                    break

        n = int((self.step_total > 0).sum())
        acc_t = (self.step_hits / self.step_total.clamp(min=1))[:n]
        alive = (self.step_total / self.step_total[0].clamp(min=1))[:n]
        self.logger.info(
            "flows alive at t  : "
            + " ".join(f"{t}:{a:.2f}" for t, a in enumerate(alive.tolist()))
        )

        extra = self._val_summary(acc_t, alive, epoch)
        extra.update(
            **{f"ep/acc_t/{t}": a for t, a in enumerate(acc_t.tolist())},
        )
        if run_tsne and all_embeddings:
            self.logger.info(
                "Computing t-SNE for validation embeddings for last 5 steps of each flow..."
            )

            x = torch.cat(all_embeddings, dim=0)
            y = torch.cat(all_targets, dim=0)

            max_samples = 200000
            if len(x) > max_samples:
                indices = torch.randperm(len(x), device=x.device)[:max_samples]
                x, y = x[indices], y[indices]

            with torch.no_grad():
                W = F.normalize(self.model.weight.detach(), p=2, dim=-1).float()
                # Create corresponding class targets for weights [0..0, 1..1, ..., 104..104]
                W_y = torch.arange(self.C, device=W.device).repeat_interleave(
                    self.model.num_sub_classes
                )
            y_np = y.cpu().numpy()
            w_y_np = W_y.cpu().numpy()
            n_samples = x.shape[0]
            x_w = torch.cat([x.float(), W], dim=0)
            x_cupy = cp.from_dlpack(x_w.float().contiguous())
            tsne = UMAP(
                metric="cosine",
                n_components=2,
                n_neighbors=15,
                random_state=42,
                output_type="numpy",
            )
            x_w_embedded = tsne.fit_transform(x_cupy)
            x_embedded = x_w_embedded[:n_samples]
            w_embedded = x_w_embedded[n_samples:]

            # fig = plt.figure(figsize=(15, 15))
            # ax = fig.add_subplot(111, projection="2d")
            fig, ax = plt.subplots(figsize=(15, 15))
            base_cmap = mpl.colormaps["gist_ncar"]

            color_list = base_cmap(np.linspace(0.0, 0.88, 105))
            custom_cmap = mcolors.ListedColormap(color_list)
            scatter = ax.scatter(
                x_embedded[:, 0],
                x_embedded[:, 1],
                c=y_np,
                marker="o",
                cmap=custom_cmap,
                s=10,
                alpha=0.5,
            )

            ax.scatter(
                w_embedded[:, 0],
                w_embedded[:, 1],
                color="black",
                marker="x",
                s=100,
                linewidths=6.0,
                alpha=0.9,
                zorder=4,
            )

            ax.scatter(
                w_embedded[:, 0],
                w_embedded[:, 1],
                c=w_y_np,
                cmap=custom_cmap,
                marker="x",
                s=100,
                linewidths=3.0,
                alpha=0.9,
                zorder=5,
            )
            x_min, x_max = np.percentile(x_embedded[:, 0], [0.1, 99.9])
            y_min, y_max = np.percentile(x_embedded[:, 1], [0.1, 99.9])
            # z_min, z_max = np.percentile(x_embedded[:, 2], [1, 99])
            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05
            # z_pad = (z_max - z_min) * 0.2

            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            # ax.set_zlim(y_min - y_pad, y_max + z_pad)
            ax.set_title(f"t-SNE 10-30 steps Embedding Clusters(Epoch {epoch})")
            fig.colorbar(scatter)

            if self.config["use_wandb"]:
                extra.update(
                    {"embedding_tsne": wandb.Image(fig, caption=f"t-SNE Epoch {epoch}")}
                )
            plt.close("all")
            del x, y, x_cupy, all_embeddings, all_targets

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        res = self.valid_metrics.result()
        res.update(extra)  # NEW
        return res

    def _progress(self, batch_idx):
        if hasattr(self.data_loader, "n_samples") and self.data_loader.n_samples:
            current = batch_idx * getattr(self.data_loader, "batch_size", 1)
            total = self.data_loader.n_samples
            unit = "samples"
        else:
            current = batch_idx
            total = getattr(self, "len_epoch", None)
            unit = "batches"

        if not total or total <= 0:
            return f"[{current} {unit}]"

        pct = 100.0 * current / total
        return f"[{current}/{total} ({pct:.0f}%)]"

    def _train_epoch_metric_update(self, step, output, flat_target, loss, plain, logit):
        self.scaler.update()
        self.writer.set_step(self.global_train_step, "train")
        self.train_metrics.update("loss", loss.item(), step=step)
        self.train_metrics.update("plain", plain.median().item(), step=step)
        self.train_metrics.update("logit", logit.median().item(), step=step)

        if flat_target.numel() > 0:
            valid_target = flat_target[flat_target >= 0]
            if valid_target.numel() > 0:
                self.dbg_metrics.update(
                    "dbg/unique_classes", valid_target.unique().numel(), step=step
                )
                self.dbg_metrics.update(
                    "dbg/top_class_share",
                    (torch.bincount(valid_target).max() / valid_target.numel()).item(),
                    step=step,
                )

        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(output, flat_target), step=step)
