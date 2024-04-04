import torch
import numpy as np
import itertools
import torch
import numpy as np

class MixedDatasetBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_sizes, dataset_ratios, batch_size, n_samples):
        
        self.dataset_sizes = dataset_sizes
        self.dataset_ratios = dataset_ratios / np.sum(dataset_ratios)

        self.batch_size = batch_size

        # Calculate number of samples from each dataset per batch
        per_batch = np.zeros(len(dataset_sizes))
        for i, dratio in enumerate(self.dataset_ratios):
            per_batch[i] = int(np.floor(dratio * batch_size))
            
        per_batch[0] = int(batch_size - np.sum(per_batch[1:]))
        self.per_batch = per_batch
        #print("BATCHES!!!!!!!!!!!!!!")
        #print(self.per_batch)
        
        self.n_samples = n_samples
        self.n_batches = n_samples // batch_size

    def __iter__(self):
        
        # Shuffle indices for each dataset
        dataset_indices = []
        tsum = 0
        for tsize, pbatch in zip(self.dataset_sizes, self.per_batch):
            tindices = tsum + (np.random.choice(tsize, int(self.n_batches * pbatch))).reshape(self.n_batches, -1)
            dataset_indices.append(tindices)
            tsum += tsize
        dataset_indices = np.concatenate(dataset_indices, axis=1).astype(int)
        #dataset_indices = [np.random.choice(size, int(self.n_batches * pbatch)).tolist() for size, pbatch in zip(self.dataset_sizes, self.per_batch)]
        
        for i in range(self.n_batches): 
            '''
            batch = []
            tsum = 0
            for tindices, pbatch in zip(dataset_indices, self.per_batch):
                #print(tsum + i * pbatch, tsum + (i + 1) * pbatch)
                batch.extend(tindices[int(tsum + i * pbatch) : int(tsum + (i + 1) * pbatch)])
                tsum += len(tindices)
            '''
            batch = dataset_indices[i].tolist()

            yield batch

    def __len__(self):
        # Total number of batches (approximate)
        return self.n_batches

