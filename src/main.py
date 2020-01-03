from robustness import imagenet_models
from robustness.attacker import AttackerModel
from utils import *
from torchvision import models 
from auxiliary_attack import gen_samples_ensemble
# from attack_target import gen_samples_with_resize

# define dataset
dataset = myData(transform=T.ToTensor())
# data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# load pre-trained models
arch = 'resnet50'
resume_path = '../models/ImageNet.pt'
if not os.path.isfile(resume_path):
    print("=> Download pretrained model...")
    os.system("wget -O ../models/ImageNet.pt")
dataset.mean = mean
dataset.std = std
classifier_model = imagenet_models.__dict__[arch](num_classes=1000)
tmp_model = AttackerModel(classifier_model, dataset)
tmp_model, _ = resume_madry_model(tmp_model, resume_path)
madry_model = InpModel(tmp_model.module.model).cuda()

# load vgg19
model_vgg1 = InpModel(models.vgg19(pretrained=True), mean, std).cuda()
models = [madry_model, model_vgg1]

# run 
gen_samples_ensemble('div_32_mean', dataset, [madry_model], [1.0], epsilon=32/255, niters=60)
gen_samples_ensemble('aux_vgg1_div_32_mean', dataset, models, [1.0, 0.1], epsilon=32/255, niters=60)


