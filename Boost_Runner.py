# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from time import time
from tqdm import tqdm

from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner


class BoostRunner(BaseRunner):
    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        # 准备日志
        logger = logging.getLogger(__name__)
        logger.info(f"Starting epoch {epoch}")

        dataset.actions_before_epoch()  # 数据预处理
        model.train()
        loss_lst = []
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
            persistent_workers=True  # 保持加载器的 worker 线程
        )

        start_time = time()
        try:
            for batch in tqdm(dl, leave=False, desc=f'Epoch {epoch:<3}', ncols=100, mininterval=1):
                batch = utils.batch_to_gpu(batch, model.device)
                model.optimizer.zero_grad()

                # 前向传播和计算损失
                out_dict = model(batch)
                loss = model.loss(out_dict)

                # 反向传播和参数更新
                loss.backward()

                # 梯度裁剪
                clip_grad_norm_(model.parameters(), max_norm=5.0)

                model.optimizer.step()
                model._update_target()

                loss_lst.append(loss.detach().cpu().item())
                del batch, out_dict  # 主动释放变量
                gc.collect()

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise

        finally:
            # 确保在异常或结束时释放资源
            gc.collect()

        avg_loss = np.mean(loss_lst).item()
        elapsed_time = time() - start_time
        logger.info(f"Epoch {epoch} completed: Avg Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
        return avg_loss
