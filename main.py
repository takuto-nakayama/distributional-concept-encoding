#  modules
from classes import General, Embedding, Density
from datetime import datetime
import argparse, time, warnings
warnings.simplefilter('ignore', FutureWarning)

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
	parser.add_argument('--gpu', type=int, default=1, nargs='?', const=True, help='0: using CPU; 1: using GPU')
	parser.add_argument('--batch', type=int, help='The number of lines that are simultaneously processed into embeddings', default=20)
	parser.add_argument('--neighbors', type=int, help='the number of dimension into which embeddings cpmpressed', default=200)
	args = parser.parse_args()

	#  starting processing
	## outputting beginning time
	start = time.time()
	t = datetime.now() 
	## general info of the process

	general = General(args.project_id, args.data_id)
	# loading data
	general.load(args.mode, args.data_source)
	## read text in a local directory or download from the input url
	info = f'''Analysis began: {t.strftime("%H:%M:%S")}.\n
-----initial settings-----
data source: {args.data_source}
project id: {args.project_id}
data id: {args.data_id}
mode: {args.mode}
tokenizer: {args.tokenizer}
model: {args.model}
gpu: {'yes' if args.gpu==1 else 'no'}
batch: {args.batch}
n_neighbors: {args.neighbors} (for UMAP)
data length: {len(general.text)} lines
'''
	general.info(info)

	#  calling the classes
	## for embedding and estimation of probability density
	embedding = Embedding(args.tokenizer, args.model, args.gpu, args.project_id, general.text)
	density = Density(args.project_id)
	
	#  embedding text in data
	embedding.embed(args.batch)
	embedding.umap(args.neighbors)
	embedding.save(args.data_id)
	t = datetime.now()
	general.subwords(embedding.dict_sw_umap)
	print(f'Subwords embedded: {t.strftime("%H:%M:%S")}.\n')

	#  estimating probability density
	density.kde(embedding.dict_sw_umap)
	t = datetime.now()
	print(f'\nKDE done: {t.strftime("%H:%M:%S")}.\n')

	density.entropy(embedding.dict_sw_umap)
	general.entropy(density.mean_entropy)
	t = datetime.now()
	print(f'entropy done: {t.strftime("%H:%M:%S")}.')

	density.save(args.data_id)

	#  terminating the proess
	end = time.time()
	t = datetime.now()
	elapsed = end - start
	print(f'''Analysis completed: {t.strftime("%H:%M:%S")}
	   Processing time: {round(elapsed, 3)} seconds.''')