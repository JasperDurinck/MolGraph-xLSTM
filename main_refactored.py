
from molgraph_xlstm.main import parse_args_dict, load_data, process_data, fit_model, test_model, set_seed
import os

def main():
    args = parse_args_dict()

    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_index']
    set_seed(10)
    
    data = load_data(args) 

    if args.get('recalc_features', False) or not ('x' in data and 'y' in data):
        data = process_data(data, args)
    
    fit_results = fit_model(data, args)
    test_results = test_model(data, args)

def get_data(args):
    data = load_data(args)
    print(data.keys())
    if args.get('recalc_features', False) or not ('x' in data and 'y' in data):
        data = process_data(data, args)
    return data

if __name__ == "__main__":
    main()
