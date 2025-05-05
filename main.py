from classes import Embedding, Density
from datetime import datetime
import argparse, time, sys


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Calculate entropy of a word in a sentence')
	parser.add_argument('data_source', type=str, help='file path or url leading to the data source')
	parser.add_argument('id', type=str, help='id to reference the data in output result')
	parser.add_argument('--mode', type=str, default='path')
	parser.add_argument('--tokenizer', type=str, help='Tokenizer to use', default='bert-base-cased')
	parser.add_argument('--model', type=str, help='Model to use', default='bert-base-cased')
	parser.add_argument('--num_samples', type=int, default=100000)
	args = parser.parse_args()

	start = time.time()
	t = datetime.now() 
	print(f'Analysis began: {t.strftime("%H:%M:%S")}.\nid: {args.id}\ndata source: {args.data_source}.')

	if args.mode == 'path':
		with open(args.data_source, 'r') as f:
			text = f.readlines()
	elif args.mode == 'url':
		t = datetime.now()
		print(f'Text loaded: {t.strftime("%H:%M:%S")}.')
	else:
		print(f'Mode Error: mode {args.mode} is invalid. "path" for loading a file on the device; "url" for a web scraping.')
		sys.exit()

	embedding = Embedding(args.tokenizer, args.model, text)
	density = Density()
	
	dict_embeddings = embedding.embed()
	embedding.save()
	t = datetime.now() 
	print(f'\n Subwords embedded: {t.strftime("%H:%M:%S")}.')

	density.kde(dict_embeddings)
	density.entropy(args.num_samples)
	density.save()
	print(f'Entropy: {density.entropy()}.')

	end = time.time()
	t = datetime.now()
	elapsed = end - start
	print(f'Analysis completed: {t.strftime("%H:%M:%S")}.\nProcessing time: {round(elapsed, 3)} seconds.')