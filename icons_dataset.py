from torch.utils.data import Dataset, DataLoader
import os
import cv2
import  numpy as np

class IconsDataset(Dataset):
    def __init__(self, datapath, split = 'val', node_id = None):
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

        self.u = u
        self.v = v
        self.n = n

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