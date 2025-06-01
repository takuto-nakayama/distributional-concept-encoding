from transformers import BertTokenizer, BertModel
from scipy.stats import gaussian_kde
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import math, torch, os, csv, sys



class General:
	def __init__(self, project_id, data_id):
		self.project_id = project_id
		self.data_id = data_id

		if 'results' not in os.dirlist:
			os.mkdir('results')
		elif f'results/{self.project_id}' not in os.dirlist('results'):
			os.mkdir(f'results/{self.project_id}')


	def info(self, info):
		if 'info' not in os.listdir(f'result/{self.project_id}'):
			os.mkdir(f'results/{self.project_id}/info')
		
		if f'{self.data_id}.txt' in os.listdir(f'results/{self.project_id}/info'):
			print('ID Error: The data id already exists in the directory. Try another.')
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



class Embedding:
	def __init__(self, tokenizer:str, model:str, project_id:str, text:list):
		self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
		self.model = BertModel.from_pretrained(model)
		self.text = text
		self.dict_sws_embs = {}


	def embed(self, batch):
		for i in range(batch,len(self.text)+1, batch):
			batch_text = self.text[i-batch:i]
			encoded = self.tokenizer(batch_text, return_tensors='pt', truncation=True, padding=True)
			subwords = [self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][i]) for i in range(encoded['input_ids'].shape[0])]
			output = self.model(**encoded).last_hidden_state.squeeze(0)

			for sws, embs in zip(subwords, output):
				for sw, emb in zip(sws, embs):
					emb = emb.unsqueeze(0)
					if sw not in self.dict_sws_embs:
						self.dict_sws_embs[sw] = emb
					else:
						self.dict_sws_embs[sw] = torch.cat((self.dict_sws_embs[sw], emb), dim=0)
			percent = int(i / len(self.text) * 100)
			process = int(percent / 2)
			print(f'\rprocessed: |{"#"*process}{"-"*(50-process)}| {percent}%', end='')
		
		return self.dict_sws_embs
	

	## Dimension reduction is required to make the number of data samples greater than the number of dimensions of each data for KDE.
	def compress(self):
		pass
	

	## write to a .hdf5 file.
	def save(self):
		pass



class Density:
	def __init__(self, project_id):
		self.dict_kde = {}
		self.list_entropy = []
		self.project_id = project_id


	def kde(self, dict_embeddings):
		for sw in dict_embeddings:
			self.dict_kde[sw] = gaussian_kde(dict_embeddings[sw].detach().numpy().T)

	
	def entropy(self, num_samples:int):
		for sw, kde in self.dict_kde.items():
			samples = kde.resample(num_samples)
			log_probs = np.log(kde.pdf(samples))
			epsilon = 1e-12
			self.list_entropy.append(-np.mean(log_probs+epsilon))
		
		self.entropy = math.mean(self.list_entropy)


	def save(self, data_id):
		if 'entropy.csv' not in os.listdir(f'results/{self.project_id}'):
			with open('entropy.csv', encoding='utf-8', mode='w') as f:
				writer = csv.writer(f)
				writer.writerow([data_id, self.entropy])
		else:
			with open('entropy.csv', encoding='utf-8', mode='a') as f:
				writer = csv.writer(f)
				writer.writerow([data_id, self.entropy])

		if 'histograms' not in os.listdir(f'results/{self.project_id}'):
			os.mkdir('histograms')

		plt.hist(self.list_entropy, bins=50)
		plt.axvline(self.entropy, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {self.entropy:.2f}")
		plt.title(f'{data_id}')
		plt.xlabel('Entropy')
		plt.ylabel('Frequency')
		plt.savefig(f'result/{self.project_id}/histograms/histogram-{data_id}.png')
		plt.show()