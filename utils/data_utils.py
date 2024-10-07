from utils.datasets import CIFAR10, CIFAR100, CIFAR100_openset, CIFAR10_openset, \
    MNIST_openset, SVHN_openset, TinyImageNet_OOD_nonoverlap, ImageNetR, VISDA, DomainNet
import json
import torch
import os
import torchvision.transforms as transforms
import numpy as np
import utils.augmix_ops as augmentations
from utils.robustbench_loader import CustomImageFolder




def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = []
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            classname = " ".join(line[1:])
            classnames.append(classname)
    return classnames


def get_desired_data(args, te_transforms, tesize=10000):
    """Return data dict(containing class names; no. of classes) and dataset.
    """
    data_dict = {}
    if args.desired == 'cifar10OOD':
        data_dict['ID_classes'] = list(json.load(open(f'datasets/cifar10_prompts_full.json')).keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        print('Test on %s level %d' %(args.corruption, args.level))
        teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
        teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
        desired_dataset = CIFAR10(root=args.dataroot,
                        train=False, download=True, transform=te_transforms)
        desired_dataset.data = teset_raw_10

    elif args.desired == 'cifar100OOD':
        data_dict['ID_classes'] = list(json.load(open(f'datasets/cifar100_prompts_full.json')).keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        print('Test on %s level %d' %(args.corruption, args.level))
        teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
        teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
        desired_dataset = CIFAR100(root=args.dataroot,
                        train=False, download=True, transform=te_transforms)
        desired_dataset.data = teset_raw_100
        
    elif args.desired == 'ImagenetROOD':

        testset = ImageNetR(root= args.dataroot)
        data_dict['ID_classes'] = list(testset.classnames)
        data_dict['ID_classes'] = ['goldfish', 'great_white_shark', 'hammerhead_shark', 'stingray', 'hen', 'ostrich', 'goldfinch', 'junco', 'bald_eagle', 'vulture', 'common_newt', 'axolotl', 'tree_frog', 'common_iguana', 'chameleon', 'Indian_cobra', 'scorpion', 'tarantula', 'centipede', 'peafowl', 'lorikeet', 'hummingbird', 'toucan', 'duck', 'goose', 'black_swan', 'koala', 'jellyfish', 'snail', 'American_lobster', 'hermit_crab', 'flamingo', 'american_egret', 'pelican', 'king_penguin', 'grey_whale', 'killer_whale', 'sea_lion', 'Chihuahua', 'shih-tzu', 'Afghan_Hound', 'Basset_Hound', 'Beagle', 'Bloodhound', 'Italian_Greyhound', 'Whippet', 'Weimaraner', 'Yorkshire_Terrier', 'Boston_Terrier', 'Scottish_Terrier', 'West_Highland_White_Terrier', 'Golden_Retriever', 'Labrador_Retriever', 'Cocker_Spaniel', 'collie', 'Border_Collie', 'Rottweiler', 'German_Shepherd_Dog', 'Boxer', 'French_Bulldog', 'saint_bernard', 'Siberian_Husky', 'Dalmatian', 'pug', 'Pomeranian', 'Chow_Chow', 'Pembroke_Welsh_Corgi', 'Toy_Poodle', 'Standard_Poodle', 'grey_wolf', 'hyena', 'red_fox', 'tabby_cat', 'leopard', 'snow_leopard', 'lion', 'tiger', 'cheetah', 'polar_bear', 'meerkat', 'ladybug', 'fly', 'bee', 'ant', 'grasshopper', 'cockroach', 'praying_mantis', 'dragonfly', 'monarch_butterfly', 'starfish', 'cottontail_rabbit', 'porcupine', 'fox_squirrel', 'beaver', 'guinea_pig', 'zebra', 'pig', 'hippopotamus', 'bison', 'gazelle', 'llama', 'skunk', 'badger', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'giant_panda', 'eel', 'anemone_fish', 'pufferfish', 'accordion', 'ambulance', 'assault_rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub', 'lighthouse', 'beer_glass', 'binoculars', 'birdhouse', 'bow_tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon', 'canoe', 'carousel', 'castle', 'mobile_phone', 'cowboy_hat', 'electric_guitar', 'fire_truck', 'flute', 'gasmask', 'grand_piano', 'guillotine', 'hammer', 'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 'lab_coat', 'lawn_mower', 'lipstick', 'mailbox', 'missile', 'mitten', 'parachute', 'pickup_truck', 'pirate_ship', 'revolver', 'rugby_ball', 'sandal', 'saxophone', 'school_bus', 'schooner', 'shield', 'soccer_ball', 'space_shuttle', 'spider_web', 'steam_locomotive', 'scarf', 'submarine', 'tank', 'tennis_ball', 'tractor', 'trombone', 'vase', 'violin', 'warplane', 'wine_bottle', 'ice_cream', 'bagel', 'pretzel', 'cheeseburger', 'hot_dog', 'cabbage', 'broccoli', 'cucumber', 'bell_pepper', 'mushroom', 'granny_smith', 'strawberry', 'lemon', 'pineapple', 'banana', 'pomegranate', 'pizza', 'burrito', 'espresso', 'volcano', 'baseball_player', 'scuba_diver', 'acorn']
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        desired_dataset = ImageNetR(root= args.dataroot, transform=te_transforms, train=True, tesize=30000)



    elif args.desired == "VisdaOOD":
        data_dict['ID_classes'] = json.load(open(f'datasets/visda_classes.json'))['classnames']
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        desired_dataset = VISDA(root= f'{args.dataroot}/visda-2017', label_files=f'datasets/visda_validation_list.txt' , transform=te_transforms, tesize=50000)
        
    elif args.desired == 'ImagenetCOOD':
        imagenet_classes = read_classnames(f'datasets/imagenet_classnames.txt')
        data_dict['ID_classes'] = imagenet_classes
        data_dict['N_classes'] = len(data_dict['ID_classes'])

        corruption_dir_path = os.path.join(args.dataroot, 'ImageNet-C/all', args.corruption,  str(args.level))
        desired_dataset = CustomImageFolder(corruption_dir_path, te_transforms, tesize=tesize)
        
    elif args.desired in ["QuickdrawOOD", "ClipartOOD", "PaintingOOD", "SketchOOD", "InfographOOD"] :
            # data_dict['ID_class_descriptions'] = json.load(open(f'{args.dataroot}/prompt_templates/domainnet_prompts_full.json'))
            data_dict['ID_classes'] = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 'table', 'teapot', 'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
            data_dict['N_classes'] = len(data_dict['ID_classes'])
            # data_dict['templates'] = cifar_templates

            img_list = {"QuickdrawOOD": 'quickdraw', "ClipartOOD":'clipart', "PaintingOOD":'painting', "SketchOOD":'sketch', "InfographOOD":'infograph'}
            desired_dataset = DomainNet(root= f'{args.dataroot}/domainnet', label_files=f'{args.dataroot}/domainnet/{img_list[args.desired]}_test.txt' , transform=te_transforms, tesize=50000)
        

    return data_dict, desired_dataset



def get_undesired_data(args, te_transforms, tesize=10000):

    data_dict = {}
    if args.undesired == 'MNIST':
        te_rize = transforms.Compose([transforms.Grayscale(3), te_transforms ])
        undesired_dataset = MNIST_openset(root=args.dataroot,
                    train=True, download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)
        data_dict['OOD_classes'] = ['digit zero', 'digit one', 'digit two', 'digit three', 'digit four', 'digit five', 'digit six', 'digit seven', 'digit eight', 'digit nine', 'digit ten']
        data_dict['OOD_classes'] = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        data_dict['N_OOD_classes'] = len(data_dict['OOD_classes'])

    
    elif args.undesired =='SVHN': 
        te_rize = transforms.Compose([te_transforms ])
        undesired_dataset = SVHN_openset(root=args.dataroot,
                    split='train', download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)
        data_dict['OOD_classes'] = ['digit zero', 'digit one', 'digit two', 'digit three', 'digit four', 'digit five', 'digit six', 'digit seven', 'digit eight', 'digit nine', 'digit ten']
        data_dict['N_OOD_classes'] = len(data_dict['OOD_classes'])

    elif args.undesired =='cifar10':
        teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
        teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
        undesired_dataset = CIFAR10_openset(root=args.dataroot,
                        train=True, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
        undesired_dataset.data = teset_raw_10[:int(tesize*args.strong_ratio)]
        data_dict['OOD_classes'] = list(json.load(open(f'datasets/cifar10_prompts_full.json')).keys())
        data_dict['N_OOD_classes'] = len(data_dict['OOD_classes'])
        

    elif args.undesired =='cifar100':
        teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
        teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
        undesired_dataset = CIFAR100_openset(root=args.dataroot,
                        train=True, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
        undesired_dataset.data = teset_raw_100[:int(tesize*args.strong_ratio)]
        data_dict['OOD_classes'] = list(json.load(open(f'datasets/cifar100_prompts_full.json')).keys())
        data_dict['N_OOD_classes'] = len(data_dict['OOD_classes'])

    elif args.undesired =='Tiny':

        transform_test = transforms.Compose([te_transforms ])
        undesired_dataset = TinyImageNet_OOD_nonoverlap(args.dataroot +'/tiny-imagenet-200', transform=transform_test, train=True)

    return data_dict, undesired_dataset


def prepare_ood_test_data(args, te_transforms):

    desired_data_dict, desired_dataset = get_desired_data(args, te_transforms, tesize=args.tesize)
    undesired_data_dict, undesired_dataset = get_undesired_data(args, te_transforms, tesize=args.tesize)
    data_dict = dict(list(desired_data_dict.items()) + list(undesired_data_dict.items()))
    id_ood_dataset = torch.utils.data.ConcatDataset([desired_dataset, undesired_dataset])
    
    ID_OOD_loader = torch.utils.data.DataLoader(id_ood_dataset, batch_size=args.batch_size, shuffle=True)

    return data_dict, id_ood_dataset, ID_OOD_loader


# TPT Transforms

# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()   # Resizing with scaling and ratio
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views
