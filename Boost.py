# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel


class Boost(GeneralModel):
    reader = 'BaseReader'
    runner = 'myRunner'
    extra_log_args = ['emb_size', 'momentum']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--momentum', type=float, default=0.995, help='Momentum update.')
        return GeneralModel.parse_model_args(parser)

    @staticmethod
    def init_weights(m):
        """Initialize weights for layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.momentum = args.momentum

        self._define_params()
        self.apply(self.init_weights)
        self._initialize_target_embeddings()

    def _define_params(self):
        """Define model parameters."""
        self.user_online = nn.Embedding(self.user_num, self.emb_size)
        self.user_target = nn.Embedding(self.user_num, self.emb_size)
        self.item_online = nn.Embedding(self.item_num, self.emb_size)
        self.item_target = nn.Embedding(self.item_num, self.emb_size)
        self.predictor = nn.Linear(self.emb_size, self.emb_size)
        self.bn = nn.BatchNorm1d(self.emb_size, eps=0, affine=False, track_running_stats=False)

    def _initialize_target_embeddings(self):
        """Initialize target embeddings as copies of online embeddings."""
        for online_param, target_param in zip(self.user_online.parameters(), self.user_target.parameters()):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False

        for online_param, target_param in zip(self.item_online.parameters(), self.item_target.parameters()):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False

    def _update_target(self):
        """Momentum update for target embeddings."""
        for online_param, target_param in zip(self.user_online.parameters(), self.user_target.parameters()):
            target_param.data.mul_(self.momentum).add_(online_param.data, alpha=1 - self.momentum)

        for online_param, target_param in zip(self.item_online.parameters(), self.item_target.parameters()):
            target_param.data.mul_(self.momentum).add_(online_param.data, alpha=1 - self.momentum)

    def forward(self, feed_dict):
        user, items = feed_dict['user_id'], feed_dict['item_id']

        # Compute predictions
        user_embeddings = self.user_online(user)
        item_embeddings = self.item_online(items)
        prediction = self._compute_prediction(user_embeddings, item_embeddings)

        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            out_dict.update(self._compute_train_embeddings(user, items))

        return out_dict

    def _compute_prediction(self, user_embeddings, item_embeddings):
        """Compute prediction scores."""
        item_proj = self.predictor(item_embeddings)
        user_proj = self.predictor(user_embeddings)

        scores = (item_proj * user_embeddings[:, None, :]).sum(dim=-1) + \
                 (user_proj[:, None, :] * item_embeddings).sum(dim=-1)
        return scores

    def _compute_train_embeddings(self, user, items):
        """Compute embeddings for training."""
        u_online = self.predictor(self.user_online(user))
        u_target = self.user_target(user)
        i_online = self.predictor(self.item_online(items).squeeze(1))
        i_target = self.item_target(items).squeeze(1)

        return {
            'u_online': u_online,
            'u_target': u_target,
            'i_online': i_online,
            'i_target': i_target
        }

    def loss(self, output):
        """Compute the loss."""
        u_online = F.normalize(output['u_online'], dim=-1)
        u_target = F.normalize(output['u_target'], dim=-1)
        i_online = F.normalize(output['i_online'], dim=-1)
        i_target = F.normalize(output['i_target'], dim=-1)

        loss_ui = 2 - 2 * (u_online * i_target.detach()).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target.detach()).sum(dim=-1)

        return (loss_ui + loss_iu).mean()

    class Dataset(GeneralModel.Dataset):
        def actions_before_epoch(self):
            """Prepare data before each epoch."""
            self.data['neg_items'] = [[] for _ in range(len(self))]
