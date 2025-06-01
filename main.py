#  modules
from classes import General, Embedding, Density
from datetime import datetime
import argparse, time, sys

#  main process
if __name__ == '__main__':
	#  parsing arguments
	parser = argparse.ArgumentParser(description='Calculate entropy of a word in a sentence')
	parser.add_argument('data_source', type=str, help='file path or url leading to the data source')
	parser.add_argument('project_id', type=str, help='id referring to the project in which output results will be in the same directory')
	parser.add_argument('data_id', type=str, help='id referring to the data that processed in the run')
	parser.add_argument('--mode', type=str, default='path')
	parser.add_argument('--tokenizer', type=str, help='Tokenizer to use', default='bert-base-cased')
	parser.add_argument('--model', type=str, help='Model to use', default='bert-base-cased')
	parser.add_argument('--batch', type=int, help='The number of lines that are simultaneously processed into embeddings', default=100)
	parser.add_argument('--num_samples', type=int, default=100000)
	args = parser.parse_args()

	#  starting processing
	## outputting beginning time
	start = time.time()
	t = datetime.now() 
	## general info of the process
	info = f'''
	Analysis began: {t.strftime("%H:%M:%S")}.
	-----initial settings-----
	data source: {args.data_source}
	project id: {args.project_id}
	data id: {args.data_id}
	mode: {args.mode}
	tokenizer: {args.tokenizer}
	model: {args.model}
	batch: {args.batch}
	sample number: {args.num_samples}
	--------------------------'''

	general = General(args.project_id, args.data_id)
	# loading data
	## read text in a local directory or download from the input url
	general.info(info)
	general.load(args.mode, args.data_source)

	#  calling the classes
	## for embedding and estimation of probability density
	embedding = Embedding(args.tokenizer, args.model, args.project_id, general.text)
	density = Density(args.project_id)
	
	#  embedding text in data
	dict_embeddings = embedding.embed(args.batch)
	embedding.save()
	t = datetime.now() 
	print(f'\n Subwords embedded: {t.strftime("%H:%M:%S")}.')

	#  estimating probability density
	density.kde(dict_embeddings)
	density.entropy(args.num_samples)
	density.save(args.data_id)
	print(f'Entropy: {density.entropy()}.')

	#  terminating the proess
	end = time.time()
	t = datetime.now()
	elapsed = end - start
	print(f'''Analysis completed: {t.strftime("%H:%M:%S")}
	   Processing time: {round(elapsed, 3)} seconds.''')