"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


def _restore_best_metrics(log_path, stop_epoch):
    """Recover checkpoint-selection baselines from an existing training log."""
    best = {"f1": float("-inf"), "map50": float("-inf"), "map5095": float("-inf")}
    best_epoch = {name: -1 for name in best}
    stage_f1 = {"stage1": float("-inf"), "stage2": float("-inf")}
    if log_path is None or not log_path.is_file():
        return best, best_epoch, stage_f1

    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            epoch = int(record["epoch"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

        detection = record.get("test_detection_metrics") or {}
        coco = record.get("test_coco_eval_bbox") or []
        values = {
            "f1": detection.get("f1"),
            "map50": coco[1] if len(coco) > 1 else None,
            "map5095": coco[0] if coco else None,
        }
        for name, value in values.items():
            if value is not None and float(value) > best[name]:
                best[name] = float(value)
                best_epoch[name] = epoch
        if values["f1"] is not None:
            stage = "stage2" if epoch >= stop_epoch else "stage1"
            stage_f1[stage] = max(stage_f1[stage], float(values["f1"]))
    return best, best_epoch, stage_f1


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        stop_epoch = self.train_dataloader.collate_fn.stop_epoch
        log_path = self.output_dir / "log.txt" if self.output_dir else None
        best, best_epoch, stage_f1 = _restore_best_metrics(
            log_path if self.last_epoch > 0 else None, stop_epoch
        )
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb,
                return_detection_metrics=True,
            )
            resume_values = {
                "f1": float(test_stats["detection_metrics"]["f1"]),
                "map50": float(test_stats["coco_eval_bbox"][1]),
                "map5095": float(test_stats["coco_eval_bbox"][0]),
            }
            for name, value in resume_values.items():
                if value > best[name]:
                    best[name] = value
                    best_epoch[name] = self.last_epoch
            stage = "stage2" if self.last_epoch >= stop_epoch else "stage1"
            stage_f1[stage] = max(stage_f1[stage], resume_values["f1"])

        print(f"Restored best metrics: {best} at epochs {best_epoch}")
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            # last.pth and periodic checkpoints are passive snapshots. They are
            # saved throughout both stages and never change the training path.
            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
                return_detection_metrics=True,
            )

            coco_stats = test_stats["coco_eval_bbox"]
            current = {
                "f1": float(test_stats["detection_metrics"]["f1"]),
                "map50": float(coco_stats[1]),
                "map5095": float(coco_stats[0]),
            }

            if self.writer and dist_utils.is_main_process():
                for i, value in enumerate(coco_stats):
                    self.writer.add_scalar(f"Test/coco_eval_bbox_{i}", value, epoch)

            checkpoint_names = {
                "f1": "best_f1_fixed.pth",
                "map50": "best_map50.pth",
                "map5095": "best_map5095.pth",
            }
            for name, value in current.items():
                if value > best[name]:
                    best[name] = value
                    best_epoch[name] = epoch
                    if self.output_dir:
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / checkpoint_names[name]
                        )

            stage = "stage2" if epoch >= stop_epoch else "stage1"
            if current["f1"] > stage_f1[stage]:
                stage_f1[stage] = current["f1"]
                if self.output_dir:
                    stage_name = "best_stg2.pth" if stage == "stage2" else "best_stg1.pth"
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / stage_name)

            print(
                "best checkpoints: "
                f"F1={best['f1']:.5f} (epoch {best_epoch['f1']}), "
                f"mAP50={best['map50']:.5f} (epoch {best_epoch['map50']}), "
                f"mAP50:95={best['map5095']:.5f} (epoch {best_epoch['map5095']})"
            )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                checkpoint_summary = {
                    "selection_rules": {
                        "best_f1_fixed.pth": "maximum validation F1 at confidence=0.5 and IoU=0.5",
                        "best_map50.pth": "maximum validation COCO AP at IoU=0.5",
                        "best_map5095.pth": "maximum validation COCO AP averaged over IoU=0.5:0.95",
                    },
                    "best": {
                        name: {"value": value, "epoch": best_epoch[name]}
                        for name, value in best.items()
                    },
                    "stage_f1": {
                        name: (None if value == float("-inf") else value)
                        for name, value in stage_f1.items()
                    },
                }
                (self.output_dir / "best_checkpoint_metrics.json").write_text(
                    json.dumps(checkpoint_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
