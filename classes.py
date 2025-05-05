from transformers import BertTokenizer, BertModel
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import math, torch, os, csv



class Embedding:
	def __init__(self, tokenizer:str, model:str, text:list):
		self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
		self.model = BertModel.from_pretrained(model)
		self.text = text
		self.dict_sws_embs = {}


	def embed(self):
		encoded = self.tokenizer(self.text, return_tensors='pt', truncation=True, padding=True)
		subwords = [self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][i]) for i in range(encoded['input_ids'].shape[0])]
		output = self.model(**encoded).last_hidden_state.squeeze(0)

		for sws, embs in zip(subwords, output):
			for sw, emb in zip(sws, embs):
				emb = emb.unsqueeze(0)
				if sw not in self.dict_sws_embs:
					self.dict_sws_embs[sw] = emb
				else:
					self.dict_sws_embs[sw] = torch.cat((self.dict_sws_embs[sw], emb), dim=0)
		
		return self.dict_sws_embs
	

	## Dimension reduction is required to make the number of data samples greater than the number of dimensions of each data for KDE.
	def compress(self):
		pass
	

	## write to a .hdf5 file.
	def save(self):
		pass



class Density:
	def __init__(self):
		self.dict_kde = {}
		self.list_entropy = []


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


	def save(self, id):
		if 'result.csv' not in os.listdir():
			with open('result.csv', encoding='utf-8', mode='w') as f:
				writer = csv.writer(f)
				writer.writerow([id, self.entropy])
		else:
			with open('result.csv', encoding='utf-8', mode='a') as f:
				writer = csv.writer(f)
				writer.writerow([id, self.entropy])

		plt.hist(self.list_entropy, bins=50)
		plt.axvline(self.entropy, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {self.entropy:.2f}")
		plt.title(f'{id}')
		plt.xlabel('Entropy')
		plt.ylabel('Frequency')
		plt.savefig(f'{id}-histogram.png')