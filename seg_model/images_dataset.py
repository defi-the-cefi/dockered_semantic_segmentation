#%%
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.utils import save_image

#%%
class images_dataset(Dataset):
    """laod image dataset."""

    def __init__(self,root_dir, min_image_size= 520, transform=True):
        """
            root_dir (string): Directory with all the images.
            transform (bool): applies image tranformations
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.list_files = os.listdir(root_dir)
        self.min_image_size = min_image_size
        print('annotating ', len(self.list_files), ' images')

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_name = self.list_files[index]
        print(img_name)
        img_path = os.path.join(self.root_dir, self.list_files[index])
        image = Image.open(img_path)

        trans_to_apply = transforms.Compose(
            [transforms.Resize(self.min_image_size,antialias=True),  # for training use transforms.RandomResizedCrop(min_image_size)
             transforms.ToTensor(),  # automagically converts all images to [0,1] values
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])  # standard procedure for image processing)
        sample = trans_to_apply(image)
        print(sample.shape)

        return sample, img_name

#%% the following calls and then inverts the class call above writing resized images to disk

def tensor_to_saved_image(dataloader, write_path):

    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])

    for x, name in dataloader:
        print(name[0])
        x = torch.squeeze(x,0)
        print(x.shape)
        x=invTrans(x)
        print(x.shape)
        # if we feed a list into save_image, make_grid is called and a grid of images is saved
        save_image(x, os.path.join(write_path,name[0]+'.jpg'))
    print('finished writing all files to disk')

if __name__ == "__main__":
    test_path = r'\images'
    test_write_path = r'\output_images'
    images = images_dataset(root_dir=test_path)
    # with batch_size = len(images.list_files) loads all images as one batch
    data_loader = DataLoader(images, batch_size=1, shuffle=False, drop_last=False)
    print(images[0])
    tensor_to_saved_image(data_loader)


