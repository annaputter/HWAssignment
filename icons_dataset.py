from torch.utils.data import Dataset, DataLoader
import os
import cv2
import  numpy as np

class IconsDataset(Dataset):
    def __init__(self, datapath, split = 'val', node_id = None, output_folder = None):
        self.images = sorted([os.path.join(datapath,file) for file in os.listdir(datapath) if file.endswith('png')])
        self.split = split

        if node_id is not None:
            self.images = [self.images[node_id]]

        self.images_size = (cv2.imread(self.images[0])).shape[0:2]
        self.images_num = len(self.images)

        # generate samples
        u,v,n = np.meshgrid(np.arange(0,self.images_size[1]),np.arange(0,self.images_size[0]),np.arange(0,self.images_num))
        u = u.reshape(-1).astype(np.int)
        v = v.reshape(-1).astype(np.int)
        n = n.reshape(-1).astype(np.int)

        # devide to val and train
        if output_folder is not None and (os.path.exists(os.path.join(output_folder,'train_split.txt')) and os.path.exists(os.path.join(output_folder,'val_split.txt'))):
            # read split from file
            if self.split == 'train':
                with open(os.path.join(output_folder, 'train_split.txt'), 'r') as f:
                    content = f.read()
            elif self.split == 'val':
                with open(os.path.join(output_folder, 'val_split.txt'), 'r') as f:
                    content = f.read()

            split_sample_idx = np.asarray([int(x) for x in content.split('\n')[:-1]])
        else:
            # generate split
            samples_idx = np.arange(0,u.shape[0])
            samples_idx = np.random.permutation(samples_idx)
            N_val = int(u.shape[0] / 100 * 20)
            samples_idx_train = samples_idx[:-N_val]
            samples_idx_val = samples_idx[-N_val:]
            if output_folder is not None:
                os.makedirs(output_folder, exist_ok=True)
                # write the split for reuse
                with open(os.path.join(output_folder, 'train_split.txt'), 'w') as output:
                    for row in samples_idx_train:
                        output.write(str(row) + '\n')
                with open(os.path.join(output_folder, 'val_split.txt'), 'w') as output:
                    for row in samples_idx_val:
                        output.write(str(row) + '\n')
            if self.split == 'train':
                split_sample_idx = samples_idx_train
            elif self.split == 'val':
                split_sample_idx = samples_idx_val

        self.u = u[split_sample_idx]
        self.v = v[split_sample_idx]
        self.n = n[split_sample_idx]

    def __len__(self):
        return self.u.size

    def get_node_id_indeces(self, node_id):
        return np.where(self.n==node_id)[0]

    def __getitem__(self, index):
        image = cv2.imread(self.images[self.n[index]])
        return self.u[index], self.v[index], self.n[index], image[self.v[index],self.u[index],:].astype(np.float32)/255.0


if __name__ == '__main__':

    datapath = './data/48/'
    dataset = IconsDataset(datapath)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=False, shuffle=False)

    for batch in dataloader:
        pass