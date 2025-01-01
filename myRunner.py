# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel

import wandb
import math
from torch.utils.data import Dataset
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
import copy

# 为了监测显存使用状况
def print_gpu_memory(stage):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[{stage}] 已分配显存: {allocated:.2f} MB, 已保留显存: {reserved:.2f} MB")
    
class myRunner(object):
	@staticmethod
	def parse_runner_args(parser):
		parser.add_argument('--epoch', type=int, default=200,
							help='Number of epochs.')
		parser.add_argument('--check_epoch', type=int, default=1,
							help='Check some tensors every check_epoch.')
		parser.add_argument('--test_epoch', type=int, default=-1,
							help='Print test results every test_epoch (-1 means no print).')
		parser.add_argument('--early_stop', type=int, default=10,
							help='The number of epochs when dev results drop continuously.')
		parser.add_argument('--lr', type=float, default=1e-3,
							help='Learning rate.')
		parser.add_argument('--l2', type=float, default=0, # 也就是weight_decay
							help='Weight decay in optimizer.')
		parser.add_argument('--batch_size', type=int, default=400, # DiffRec的batch_size是400
							help='Batch size during training.')
		parser.add_argument('--eval_batch_size', type=int, default=400, # DiffRec的eval和train用一样的batch_size
							help='Batch size during testing.')
		parser.add_argument('--optimizer', type=str, default='AdamW', # DiffRec使用的是AdamW，尽管默认weight_decay是0
							help='optimizer: SGD, Adam, Adagrad, Adadelta, AdamW')
		parser.add_argument('--num_workers', type=int, default=4, # DiffRec的num_workers是4
							help='Number of processors when prepare batches in DataLoader')
		parser.add_argument('--pin_memory', type=int, default=1, # DiffRec的pin_memory是1
							help='pin_memory in DataLoader')
		parser.add_argument('--topk', type=str, default='5,10,20,50',
							help='The number of items recommended to each user.')
		parser.add_argument('--metric', type=str, default='NDCG,HR',
							help='metrics: NDCG, HR')
		parser.add_argument('--main_metric', type=str, default='',
							help='Main metric to determine the best model.')
		# 修改为L-DiffRec中的参数,只有启用了L-DiffRec才会用到这些参数
		parser.add_argument('--LDiffRec', action='store_true', help='train L-DiffRec model or LT-DiffRec model')
		parser.add_argument('--lr1', type=float, default=0.0001, help='learning rate for Autoencoder')
		parser.add_argument('--lr2', type=float, default=0.0001, help='learning rate for MLP')
		parser.add_argument('--wd1', type=float, default=0.0, help='weight decay for Autoencoder')
		parser.add_argument('--wd2', type=float, default=0.0, help='weight decay for MLP')
		parser.add_argument('--optimizer1', type=str, default='AdamW', help='optimizer for AE: Adam, AdamW, SGD, Adagrad, Momentum')
		parser.add_argument('--optimizer2', type=str, default='AdamW', help='optimizer for MLP: Adam, AdamW, SGD, Adagrad, Momentum')
		return parser

	@staticmethod
	def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
		"""
		:param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
		:param topk: top-K value list
		:param metrics: metric string list
		:return: a result dict, the keys are metric@topk
		"""
		evaluations = dict()
		# sort_idx = (-predictions).argsort(axis=1)
		# gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
		# ↓ As we only have one positive sample, comparing with the first item will be more efficient. 
		gt_rank = (predictions >= predictions[:,0].reshape(-1,1)).sum(axis=-1)
		# if (gt_rank!=1).mean()<=0.05: # maybe all predictions are the same
		# 	predictions_rnd = predictions.copy()
		# 	predictions_rnd[:,1:] += np.random.rand(predictions_rnd.shape[0], predictions_rnd.shape[1]-1)*1e-6
		# 	gt_rank = (predictions_rnd > predictions[:,0].reshape(-1,1)).sum(axis=-1)+1
		for k in topk:
			hit = (gt_rank <= k)
			for metric in metrics:
				key = '{}@{}'.format(metric, k)
				if metric == 'HR':
					evaluations[key] = hit.mean()
				elif metric == 'NDCG':
					evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
				else:
					raise ValueError('Undefined evaluation metric: {}.'.format(metric))
		# 检查显存
		print_gpu_memory("evaluate_method")
		return evaluations

	def computeTopNAccuracy(self, GroundTruth, predictedIndices, topN):
		'''
			DiffRec提供的评估方法
		'''
		precision = [] 
		recall = [] 
		NDCG = [] 
		MRR = []
		
		for index in range(len(topN)):
			sumForPrecision = 0
			sumForRecall = 0
			sumForNdcg = 0
			sumForMRR = 0
			for i in range(len(predictedIndices)):
				if len(GroundTruth[i]) != 0:
					mrrFlag = True
					userHit = 0
					userMRR = 0
					dcg = 0
					idcg = 0
					idcgCount = len(GroundTruth[i])
					ndcg = 0
					hit = []
					for j in range(topN[index]):
						if predictedIndices[i][j] in GroundTruth[i]:
							# if Hit!
							dcg += 1.0/math.log2(j + 2)
							if mrrFlag:
								userMRR = (1.0/(j+1.0))
								mrrFlag = False
							userHit += 1
					
						if idcgCount > 0:
							idcg += 1.0/math.log2(j + 2)
							idcgCount = idcgCount-1
								
					if(idcg != 0):
						ndcg += (dcg/idcg)
						
					sumForPrecision += userHit / topN[index]
					sumForRecall += userHit / len(GroundTruth[i])               
					sumForNdcg += ndcg
					sumForMRR += userMRR

			precision.append(round(sumForPrecision / len(predictedIndices), 4))
			recall.append(round(sumForRecall / len(predictedIndices), 4))
			NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
			MRR.append(round(sumForMRR / len(predictedIndices), 4))
		
		evaluations = dict()
		for k in topN:
			for metric in ['Precision', 'Recall', 'NDCG', 'MRR']:
				key = '{}@{}'.format(metric, k)
				if metric == 'Precision':
					evaluations[key] = precision[topN.index(k)]
				elif metric == 'Recall':
					evaluations[key] = recall[topN.index(k)]
				elif metric == 'NDCG':
					evaluations[key] = NDCG[topN.index(k)]
				elif metric == 'MRR':
					evaluations[key] = MRR[topN.index(k)]
		return evaluations


	def __init__(self, args):
		self.train_models = args.train
		self.epoch = args.epoch
		self.check_epoch = args.check_epoch
		self.test_epoch = args.test_epoch
		self.early_stop = args.early_stop
		self.learning_rate = args.lr
		self.batch_size = args.batch_size
		self.eval_batch_size = args.eval_batch_size
		self.l2 = args.l2
		self.optimizer_name = args.optimizer
		self.num_workers = args.num_workers
		self.pin_memory = args.pin_memory
		self.topk = [int(x) for x in args.topk.split(',')]
		self.metrics = [m.strip().upper() for m in args.metric.split(',')]
		self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0]) if not len(args.main_metric) else args.main_metric # early stop based on main_metric
		self.main_topk = int(self.main_metric.split("@")[1])
		self.time = None  # will store [start_time, last_step_time]
		self.random_seed = args.random_seed
		self.log_path = os.path.dirname(args.log_file) # path to save predictions
		self.save_appendix = args.log_file.split("/")[-1].split(".")[0] # appendix for prediction saving
		if args.LDiffRec:
			self.LDiffRec = True
			self.optimizer1 = args.optimizer1
			self.optimizer2 = args.optimizer2
			self.lr1 = args.lr1
			self.lr2 = args.lr2
			self.wd1 = args.wd1
			self.wd2 = args.wd2
		# 检查显存
		print_gpu_memory("runner初始化")

	def _check_time(self, start=False):
		if self.time is None or start:
			self.time = [time()] * 2
			return self.time[0]
		tmp_time = self.time[1]
		self.time[1] = time()
		return self.time[1] - tmp_time

	def _build_optimizer(self, model):
		if self.LDiffRec:
			logging.info('Optimizer1: ' + self.optimizer1)
			if self.optimizer1 == 'Adagrad':
				optimizer1 = eval('torch.optim.{}'.format(self.optimizer1))(
					model.Autoencoder.parameters(), lr=self.lr1, weight_decay=self.wd1, initial_accumulator_value=1e-8,)
			elif self.optimizer1 == 'Momentum':
				optimizer1 = eval('torch.optim.{}'.format(self.optimizer1))(
					model.Autoencoder.parameters(), lr=self.lr1, weight_decay=self.wd1, momentum=0.95,)
			else:
				optimizer1 = eval('torch.optim.{}'.format(self.optimizer1))(
					model.Autoencoder.parameters(), lr=self.lr1, weight_decay=self.wd1)
			
			logging.info('Optimizer2: ' + self.optimizer2)
			if self.optimizer2 == 'Adagrad':
				optimizer2 = eval('torch.optim.{}'.format(self.optimizer2))(
					model.DNN.parameters(), lr=self.lr2, weight_decay=self.wd2, initial_accumulator_value=1e-8,)
			elif self.optimizer2 == 'Momentum':
				optimizer2 = eval('torch.optim.{}'.format(self.optimizer2))(
					model.DNN.parameters(), lr=self.lr2, weight_decay=self.wd2, momentum=0.95,)
			else:
				optimizer2 = eval('torch.optim.{}'.format(self.optimizer2))(
					model.DNN.parameters(), lr=self.lr2, weight_decay=self.wd2)
			return optimizer1, optimizer2
		else:
			logging.info('Optimizer: ' + self.optimizer_name)
			optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
				model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
			return optimizer

	def train(self, data_dict: Dict[str, BaseModel.Dataset]):
		model = data_dict['train'].model
		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)
		utils.init_seed(self.random_seed)
		try:
			for epoch in range(self.epoch):
				# Fit
				self._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				loss = self.fit(data_dict['train'], epoch=epoch + 1)
				# 检查显存
				print_gpu_memory("Train: Epoch {}".format(epoch+1))
    
				if np.isnan(loss):
					logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
					break
				training_time = self._check_time()
				# Observe selected tensors DiffRec没有这个check_list
				if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
					utils.check(model.check_list)
	 
				if 'DIFFREC' in self.metrics:
					# Record dev results
					dev_result = self.evaluate(data_dict['train'], data_dict['dev'], mask_his=[data_dict['train']], topks=[self.main_topk], metrics=['DIFFREC'])
					dev_results.append(dev_result)
					main_metric_results.append(dev_result[self.main_metric])
					logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	dev=({})'.format(
						epoch + 1, loss, training_time, utils.format_metric(dev_result))

		
					# Test
					if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
						test_result = self.evaluate(data_dict['train'], data_dict['test'], mask_his=[data_dict['train']], topks=self.topk[:1], metrics=['DIFFREC'])
						logging_str += ' test=({})'.format(utils.format_metric(test_result))
					testing_time = self._check_time()
					logging_str += ' [{:<.1f} s]'.format(testing_time)
				else:
					# Record dev results
					dev_result = self.evaluate(data_dict['train'], data_dict['dev'], [self.main_topk], self.metrics)
					dev_results.append(dev_result)
					main_metric_results.append(dev_result[self.main_metric])
					logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	dev=({})'.format(
						epoch + 1, loss, training_time, utils.format_metric(dev_result))
					log_result = copy.deepcopy(dev_result)
					log_result = log_result.update({"loss": loss, "epoch": epoch})
					wandb.log(dev_result)
					# Test
					if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
						test_result = self.evaluate(data_dict['train'], data_dict['test'], self.topk[:1], self.metrics)
						logging_str += ' test=({})'.format(utils.format_metric(test_result))
					testing_time = self._check_time()
					logging_str += ' [{:<.1f} s]'.format(testing_time)

				# Save model and early stop
				if max(main_metric_results) == main_metric_results[-1] or \
						(hasattr(model, 'stage') and model.stage == 1):
					best_epoch = epoch
					model.save_model()
					logging_str += ' *'
				logging.info(logging_str)

				if self.early_stop > 0 and self.eval_termination(main_metric_results):
					logging.info("Early stop at %d based on dev result." % (epoch + 1))
					break

		except KeyboardInterrupt:
			logging.info("Early stop manually")
			exit_here = input("Exit completely without evaluation? (y/n) (default n):")
			if exit_here.lower().startswith('y'):
				logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
				exit(1)

		# Find the best dev result across iterations
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
		model.load_model()

	def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
		# Random seed
		def worker_init_fn(worker_id):
			np.random.seed(self.random_seed + worker_id)

		class CustomBatchSampler(BatchSampler):
			def __iter__(self):
				batch_indices = list(super().__iter__())
				for batch in batch_indices:
					print("Batch indices:", batch)
					yield batch

		model = dataset.model
		if model.optimizer is None and (model.optimizer1 is None or model.optimizer2 is None):
			if self.LDiffRec:
				model.optimizer1, model.optimizer2 = self._build_optimizer(model)
			else:
				model.optimizer = self._build_optimizer(model)
		dataset.actions_before_epoch()  # must sample before multi thread start
		model.train()
		loss_lst = list()
		# 自定义 BatchSampler
		# batch_sampler = CustomBatchSampler(RandomSampler(dataset), batch_size=self.batch_size, drop_last=False)
		# dl = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=self.num_workers,
		# 				pin_memory=self.pin_memory, worker_init_fn=worker_init_fn)
		dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						pin_memory=self.pin_memory, worker_init_fn=worker_init_fn)
		# 使用train_dataset作为模型的输入,loss是由GaussianDiffusion.training_losses计算的
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			if self.LDiffRec:
				model.optimizer1.zero_grad()
				model.optimizer2.zero_grad()
    
				loss = model.loss(batch.to(model.device))
    
				model.update_count_vae += 1
				loss.backward()
				model.optimizer1.step()
				model.optimizer2.step()
				model.update_count += 1
				loss_lst.append(loss.detach().cpu().data.numpy())
			else:
				model.optimizer.zero_grad()
				loss = model.loss(batch.to(model.device))
				loss.backward()
				model.optimizer.step()
				loss_lst.append(loss.detach().cpu().data.numpy())

		return np.sum(loss_lst).item()

	def eval_termination(self, criterion: List[float]) -> bool:
		if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
			return True
		elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
			return True
		return False

	def evaluate(self,
		train_dataset:BaseModel.Dataset,
		test_dataset: BaseModel.Dataset,
		topks: list,
		metrics: list,
		mask_his:list[BaseModel.Dataset]|None = None,
		) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		if "DIFFREC" in metrics:
			assert mask_his is not None, "mask_his should be provided for DIFFREC"
			# 整理 target_items
			target_items = []
			target_users = []
			for i in range(len(test_dataset)):
				if len(test_dataset.csr_matrix[i, :].nonzero()[1].tolist()) > 0:
					target_items.append(test_dataset.csr_matrix[i, :].nonzero()[1].tolist())
					target_users.append(i)

			# 合并所有历史交互的稀疏矩阵
			mask_his_csr_matrix = mask_his[0].csr_matrix
			for mask in mask_his[1:]:
				mask_his_csr_matrix += mask.csr_matrix

			# # 将用户已经交互过的物品的预测值设置为负无穷
			# for prediction in predictions:
			# 	prediction[mask_his_csr_matrix.nonzero()] = -np.inf
	
			# train_dataset需要作为模型的输入，然后第二个dataset是用来挑选要预测的user_id
			predict_items = self.predict(train_dataset,target_users,mask_his_csr_matrix,topks) # 掩码后的预测结果 (n_users, n_items)

			# # 整理预测结果
			# predict_items = []
			# for i in range(len(predictions)):
			# 	_, indices = torch.topk(predictions[i], topks[-1])
			# 	predict_items.append(indices.cpu().numpy().tolist())
			return self.computeTopNAccuracy(target_items, predict_items, topks)
		else: 
			# Rechorus的评估方法
			target_items = []
			neg_items = []
			target_users = []
			for i in range(len(test_dataset.data['user_id'])):
				target_items.append(test_dataset.data['item_id'][i])
				neg_items.append(test_dataset.data['neg_items'][i])
				target_users.append(test_dataset.data['user_id'][i])
			
			# 得到target_users的所有物品评分，
			predict_all_items = self.predict(train_dataset, target_users)
			# users_id:在predict_all_items中的index，也就是user_id在set(target_users)中大小排序
			index_map = {}
			target_users_sorted = sorted(set(target_users))
			for i, user_id in enumerate(target_users_sorted):
				index_map[user_id] = i
			
			# 整理预测结果，按照测试集的user_id的顺序排序和每行指定的neg_items来筛选
			predictions = list()
			for i in range(len(target_users)):
				prediction = predict_all_items[index_map[target_users[i]]][[target_items[i]] + neg_items[i]]
				predictions.append(prediction)
				# 输出每个用户的预测结果
				# print(f"User {target_users[i]}: Target item {target_items[i]}, Negative items {neg_items[i]}")
				# print("Prediction:", prediction)
			# 检查显存
			print_gpu_memory("整理预测结果形状为(-1,n_candidates)")
			predictions = np.array(predictions)
			# print("Predictions shape:", predictions.shape)
			# print("Predictions array:", predictions)
			return self.evaluate_method(predictions, topks, metrics)

	def predict(self,
		train_dataset:BaseModel.Dataset,
		target_users=None, #可重复可不重复
		mask_csr_matrix=None,
		topks=None,
		) -> np.ndarray:
		with torch.no_grad():
			train_dataset.model.eval()
			predictions = list()
			predict_items = list()
			# 从训练数据作为模型输入，对每个用户的所有候选集进行预测，然后再去掉已经点击过的，来看测试集的排名
			if target_users is not None:
				dataset = FilteredDataset(train_dataset, target_users) # 使用筛选后的数据集，只会取到target_users的训练数据
			else:
				dataset = train_dataset
			dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, pin_memory=self.pin_memory) #按id从小到大顺序取数据
			for batch_idx, batch in enumerate(tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict')):
				gc.collect()
				torch.cuda.empty_cache()
				prediction = train_dataset.model(batch.to(train_dataset.model.device))['prediction']
				if mask_csr_matrix is not None: # 预测值设置为负无穷
					prediction[mask_csr_matrix[batch_idx*self.batch_size:batch_idx*self.batch_size+len(batch)].nonzero()] = -np.inf
				if topks is not None:
					_, indices = torch.topk(prediction, topks[-1])
					predict_items.extend(indices.cpu().numpy().tolist())
				else: # 没有要求k，那就返回全部评分
					predictions.extend(prediction.cpu().data.numpy())
				# 检查显存
				print_gpu_memory("predict:{}/{}".format(batch_idx+1, len(dl)))

			if topks is not None:
				return predict_items
			else:
				predictions = np.array(predictions)
				# 检查显存
				print_gpu_memory("把predictions转换成numpy数组")
				return predictions

	def print_res(self, train_dataset:BaseModel.Dataset, dataset: BaseModel.Dataset,mask_his:list[BaseModel.Dataset]|None = None, wandb:bool=False) -> str:
		"""
		Construct the final result string before/after training
		:return: test result string
		DiffRec需要传入train_dataset
		"""
		if 'DIFFREC' in self.metrics:
			result_dict = self.evaluate(train_dataset, dataset, mask_his=mask_his, topks=self.topk, metrics=['DIFFREC'])
		else:
			result_dict = self.evaluate(train_dataset, dataset, topks=self.topk, metrics=self.metrics)
		if wandb:
			return result_dict
		res_str = '(' + utils.format_metric(result_dict) + ')'
		return res_str


# 用target_users来过滤数据集
class FilteredDataset(Dataset):
	def __init__(self, dataset, target_users):
		self.dataset = dataset
		self.target_users = set(target_users)  # 使用集合提高查找效率
		self.filtered_indices = [i for i in range(len(dataset)) if i in self.target_users]

	def __len__(self):
		return len(self.filtered_indices)

	def __getitem__(self, idx):
		actual_idx = self.filtered_indices[idx]
		return self.dataset[actual_idx]