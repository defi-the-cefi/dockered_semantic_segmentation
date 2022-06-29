#%%
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
# requires images of min size: 520
from images_dataset import images_dataset, tensor_to_saved_image
from torch.utils.data import DataLoader
import json

read_path = r'./images'
image_write_path = r'./output_images'
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

img_dataset = images_dataset(root_dir=read_path)
img_loader = DataLoader(img_dataset, batch_size=1, shuffle=False, drop_last=False)
tensor_to_saved_image(img_loader, image_write_path)
model = deeplabv3_resnet50(pretrained=True, progress=True).to(device)
model = model.eval()

def model_inference(input):
    print('input image shape ', input.shape)
    with torch.no_grad():
        #model.eval()
        outputs = model(input.to(device))['out']
        print('forward pass run on input batch')
        print('output image shape: ', outputs.shape)
        return(outputs)

def batched_inferences():
    annotations = {}
    for img,img_name in img_loader:
        # batching all the images to perform in one batch can be done with torch.stack
        print('annotating ', img_name)
        model_output = model_inference(img)
        print('annotations generated appending to dict ')
        #print(outputs.shape, outputs.min().item(), outputs.max().item())
        for index, item in enumerate(img_name):
            print('appending annotations for: ', item)
            print('index: ', index)
            annotations[item] = model_output.select(0,index)
            torch.save('./output_images', item+'.pth')
    # print(annotations)
    return(annotations)

final_outputs = batched_inferences()
print('ready for prost processing and viz')

# with open("annotations.json", "w") as fp:
#     json.dump(annotations, fp)
