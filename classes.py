from transformers import BertTokenizer, BertModel
from scipy.stats import gaussian_kde
from datetime import datetime
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import math, torch, os, csv, sys, statistics, h5py



class General:
	def __init__(self, project_id, data_id):
		self.project_id = project_id
		self.data_id = data_id

		if 'results' not in os.listdir():
			os.mkdir('results')
			os.mkdir(f'results/{self.project_id}')
		elif self.project_id not in os.listdir('results'):
			os.mkdir(f'results/{self.project_id}')


	def info(self, info):
		if 'info' not in os.listdir(f'results/{self.project_id}'):
			os.mkdir(f'results/{self.project_id}/info')
		
		if f'info-{self.data_id}.txt' in os.listdir(f'results/{self.project_id}/info'):
			print('IdError: The data id already exists in the directory. Try another.')
			sys.exit()
		
		with open(f'results/{self.project_id}/info/info-{self.data_id}.txt', 'w') as f:
			f.write(info)

		print(info)
	

	def load(self, mode, data_source):
		if mode == 'path':
			with open(data_source, 'r') as f:
				self.text = f.readlines()
		elif mode == 'url':
			t = datetime.now()
			print(f'Text loaded: {t.strftime("%H:%M:%S")}.')
		else:
			print(f'Mode Error: mode \"{mode}\" is invalid. "path" for loading a file on the device; "url" for a web scraping.')
			sys.exit()


	def subwords(self, dict_embeddings):
		with open(f'results/{self.project_id}/info/info-{self.data_id}.txt', 'a') as f:
			f.write('\n\n------result summary------\n')
			f.write(f'data amount: {len(dict_embeddings)} subwords')
		print(f'\nData amount: {len(dict_embeddings)} subwords.')


	def entropy(self, entropy):
		with open(f'results/{self.project_id}/info/info-{self.data_id}.txt', 'a') as f:
			f.write(f'\nentropy: {entropy}')
		print(f'\nentropy: {entropy}')



class Embedding:
	def __init__(self, tokenizer:str, model:str, gpu:int, project_id:str, text:list):
		self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
		self.model = BertModel.from_pretrained(model)
		self.gpu = gpu
		self.project_id = project_id
		self.text = text
		self.dict_sws_embs = defaultdict(list)

		if self.gpu == 1:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			self.model = self.model.to(self.device)
			print(f'Using device: {self.device}\n')
		else:
			print(f'Using device: cpu\n')


	def embed(self, batch):
		for i in range(batch,len(self.text)+batch, batch):
			batch_text = self.text[i-batch:min((i, len(self.text)))]
			encoded = self.tokenizer(batch_text, return_tensors='pt', truncation=True, padding=True)
			if self.gpu == 1:
				encoded = {key: value.to(self.device) for key, value in encoded.items()}
			subwords = [self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][i]) for i in range(encoded['input_ids'].shape[0])]
			with torch.no_grad():
				output = self.model(**encoded).last_hidden_state.squeeze(0)
				for sws, embs in zip(subwords, output):
					for sw, emb in zip(sws, embs):
						self.dict_sws_embs[sw].append(emb.cpu())

			percent = int(i / len(self.text) * 100)
			process = int(percent / 2)
			print(f'\rembedding: |{"#"*process}{"-"*(50-process)}| {percent}% ({min((i,len(self.text)))}/{len(self.text)})', end='')

		for sw in self.dict_sws_embs:
			self.dict_sws_embs[sw] = torch.stack(self.dict_sws_embs[sw])  
		

	## Dimension reduction is required to make the number of data samples greater than the number of dimensions of each data for KDE.
	def pca(self, d):
		self.dict_sw_pca = {}
		pca = PCA(n_components=d)
		
		for sw in self.dict_sws_embs:
			try:
				self.dict_sw_pca[sw] = pca.fit_transform(self.dict_sws_embs[sw])
			except:
				continue

	## write to a .hdf5 file.
	def save(self, data_id):
		if f'emb-{self.project_id}.hdf5' not in os.listdir(f'results/{self.project_id}'):
			with h5py.File(f'results/{self.project_id}/emb-{data_id}.hdf5', 'w') as f:
				group = f.create_group(data_id)
				for sw, emb in self.dict_sw_pca.items():
					group.create_dataset(sw, data=emb)
		else:
			with h5py.File(f'results/{self.project_id}/emb-{data_id}.hdf5', 'a') as f:
				group = f.create_group(data_id)
				for sw, emb in self.dict_sw_pca.items():
					group.create_dataset(sw, data=emb)



class Density:
	def __init__(self, project_id):
		self.dict_kde = {}
		self.list_entropy = []
		self.project_id = project_id


	def kde(self, dict_embeddings):
		cnt = 0
		for sw in dict_embeddings:
			try:
				self.dict_kde[sw] = gaussian_kde(dict_embeddings[sw].T)
			except ValueError:
				pass
			cnt += 1
			percent = int(cnt / len(dict_embeddings) * 100)
			process = int(percent / 2)
			print(f'\rkde: |{"#"*process}{"-"*(50-process)}| {percent}% ({cnt}/{len(dict_embeddings)})', end='')

	
	def entropy(self, num_samples:int):
		cnt = 0
		for sw, kde in self.dict_kde.items():
			samples = kde.resample(num_samples)
			pdf_vals = kde.pdf(samples)
			pdf_vals = np.clip(pdf_vals, 1e-12, None)
			log_probs = np.log2(pdf_vals)
			self.list_entropy.append(-np.mean(log_probs))

			cnt += 1
			percent = int(cnt / len(self.dict_kde) * 100)
			process = int(percent / 2)
			print(f'\rentropy: |{"#"*process}{"-"*(50-process)}| {percent}% ({cnt}/{len(self.dict_kde)})', end='')

		self.mean_entropy = statistics.mean(self.list_entropy)


	def save(self, data_id):
		## save the mean of entropies of each subword
		if 'entropy.csv' not in os.listdir(f'results/{self.project_id}'):
			with open(f'results/{self.project_id}/entropy.csv', 'w', encoding='utf-8') as f:
				writer = csv.writer(f)
				writer.writerow([data_id, self.mean_entropy])
		else:
			with open(f'results/{self.project_id}/entropy.csv', 'a', encoding='utf-8') as f:
				writer = csv.writer(f)
				writer.writerow([data_id, self.mean_entropy])

		## save frequency distribution
		rank = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]
		fd = []
		min = 0
		for r in rank:
			fd.append(sum(x <= r for x in self.list_entropy) - sum(x <= min for x in self.list_entropy))
			min = r
		fd.append(sum(x > 10.0 for x in self.list_entropy))

		if 'frequency-distribution.csv' not in os.listdir(f'results/{self.project_id}'):
			with open(f'results/{self.project_id}/frequency-distribtion.csv', encoding='utf-8', mode='w') as f:
				writer = csv.writer(f)
				writer.writerow([''] + rank + ['10.0 <'])
				writer.writerow([f'{data_id}'] + fd)
		else:
			with open(f'results/{self.project_id}/frequency-distribtion.csv', encoding='utf-8', mode='a') as f:
				writer = csv.writer(f)
				writer.writerow([f'{data_id}'] + fd)

		## plot histograms of entropy frequency
		if 'histograms' not in os.listdir(f'results/{self.project_id}'):
			os.mkdir(f'results/{self.project_id}/histograms')

		plt.hist(self.list_entropy, bins=50)
		plt.axvline(self.mean_entropy, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {self.mean_entropy:.2f}")
		plt.title(f'{data_id}')
		plt.xlabel('Entropy')
		plt.ylabel('Frequency')
		plt.savefig(f'results/{self.project_id}/histograms/histogram-{data_id}.png')