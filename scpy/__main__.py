import sys, os
import numpy as np
import torch
import importlib.util
import argparse
import pandas as pd
from threading import Thread

from .functional import fit_batch, _parallel, Dataset, dataloader, dataserver


# parse args

parser = argparse.ArgumentParser()

parser.add_argument('input_py', type=str, help='A python scripy provides the data used for fitting and default parameters.')
parser.add_argument('output_csv', type=str, help='A csv file that saves the results. The file with the same name will be overwritten.')
parser.add_argument('-d', '--devices', type=str, help='Torch devices to be used, separated by commas.')
parser.add_argument('--disp', type=str, default='True', help='Whether to show the progress bar.')
parser.add_argument('--low_memory', type=str, default='True', help='If False, the data required by each process will be pre-transmitted.')

args = parser.parse_args()

import_spec = importlib.util.spec_from_file_location('input_py', args.input_py)
input_py = importlib.util.module_from_spec(import_spec)
import_spec.loader.exec_module(input_py)

X = input_py.args['X']
Y = input_py.args['Y']
family = input_py.args['family']
xnames = input_py.args.get('xnames')
yname = input_py.args.get('yname')
disp = args.disp.lower() == 'true'
low_memory = args.low_memory.lower() == 'true'
kwargs = input_py.args['kwargs'] if 'kwargs' in input_py.args else {}

if args.devices is None:
    devices = input_py.args['devices']
else:
    devices = args.devices.split(',')
    devices = [(device if device == 'cpu' else int(device)) for device in devices]

# check args

devices = list(set(devices))
assert len(devices) > 0, 'The devices list is empty'

# fit
import time

if len(devices) == 1:
  
    if type(Y) == dict:
        Y = dataloader(**Y)
        
    tick = time.time()
    results = fit_batch(X, Y, family, devices[0], xnames, yname, **kwargs)
    print(time.time() - tick)
    
else:
  
    ctx = torch.multiprocessing.get_context("spawn")
    
    # split args
  
    if type(Y) == dict:
      
        length = Y['length'] if 'length' in Y else len(Y['data'])
        batch_size = int(np.ceil(length / float(len(devices))))
        loader = Y['data']
        
    else:
        
        length = len(Y)
        batch_size = int(np.ceil(length / float(len(devices))))
        loader = Y.__getitem__

    if low_memory:
        
        servers, clients = zip(*map(ctx.Pipe, [True] * len(devices)))

        Ys = [{'data': client, 'length': min(batch_size, length - batch_size * i), 'offset': batch_size * i} for i, client in enumerate(clients)]

        for server in servers:
            Thread(target=dataserver, args=(server, loader)).start()
            
    else:
      
        if type(Y) == dict:
            Y = Dataset(loader, length)
        
        Ys = [Y[i:i+batch_size] for i in range(0, length, batch_size)]

    if yname is None:
        yname = range(length)
    else:
        assert length == len(yname), 'The batch size of Y is different from yname: {} != {}'.format(length, len(yname))
        
    ynames = [yname[i:i+batch_size] for i in range(0, length, batch_size)]
        
    # fit

    pool = ctx.Pool(len(devices))
    
    tick = time.time()
    results = list(pool.starmap(_parallel, zip([X] * len(devices), Ys, [family] * len(devices), devices, [xnames] * len(devices), ynames, [disp] * len(devices), [kwargs] * len(devices))))
    print(time.time() - tick)
    
    for client in clients:
        client.send(None)
    
    # finish

    results = pd.concat(results, axis=0)
  
results.to_csv(args.output_csv)
print(results)

  