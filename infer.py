import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os


parser = argparse.ArgumentParser(description='Class Selective Loss for Partial Multi Label Classification.')

parser.add_argument('--model_path', type=str, default='./pretrained_weights/flixstock_ignore_best_model-5-1250.ckpt')       # default is './models/mtresnet_opim_86.72.pth'
parser.add_argument('--pic_path', type=str, default='../Datasets/FlixstockTask/images_split/test/images/2e1cbe2c-6267-4e16-8938-c4e02af0b4ac1536662609762-HERENOW-Men-Tshirts-2871536662608198-2.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_m')          # keep the same
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dataset_type', type=str, default='flixstock')      # default is OpenImagess
parser.add_argument('--class_description_path', type=str, default='./data/oidv6-class-descriptions.csv')
parser.add_argument('--th', type=float, default=0.97)
parser.add_argument('--top_k', type=float, default=3)
parser.add_argument('--num-classes', default=21)                                # Flixstock: 21
parser.add_argument('--all_val_inference', action='store_true')                # to do inference on all test images
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='../Datasets/FlixstockTask/')


def inference(im, model, class_list, args):
    if isinstance(im, str):
        im = Image.open(im)
    im_resize = im.resize((args.input_size, args.input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()

    # print('All prediction scores:')
    # print(np_output)

    if args.dataset_type.lower() == 'flixstock':
        best_pred_neck_value = np.max(np_output[:7])
        best_pred_neck_index = np.argmax(np_output[:7])
        best_pred_neck_type = np.array(class_list)[:7][best_pred_neck_index]

        best_pred_sleeve_length_value = np.max(np_output[7:11])
        best_pred_sleeve_length_index = np.argmax(np_output[7:11])
        best_pred_sleeve_length_type = np.array(class_list)[7:11][best_pred_sleeve_length_index]

        best_pred_pattern_value = np.max(np_output[11:])
        best_pred_pattern_index = np.argmax(np_output[11:])
        best_pred_pattern_type = np.array(class_list)[11:][best_pred_pattern_index]

        detected_classes = np.array([best_pred_neck_type, best_pred_sleeve_length_type, best_pred_pattern_type])
        detected_class_indexes = np.array([best_pred_neck_index, best_pred_sleeve_length_index, best_pred_pattern_index])
        scores = np.array([best_pred_neck_value, best_pred_sleeve_length_value, best_pred_pattern_value])

        return detected_classes, detected_class_indexes, scores, im

    else:
        idx_sort = np.argsort(-np_output)
        # Top-k
        detected_classes = np.array(class_list)[idx_sort][: args.top_k]
        scores = np_output[idx_sort][: args.top_k]
        # Threshold
        idx_th = scores > args.th
        return detected_classes[idx_th], scores[idx_th], im


def display_image(im, tags, filename):

    path_dest = "./results"
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    # plt.rcParams["axes.titlesize"] = 10
    plt.title("Predicted classes: {}".format(tags))
    plt.savefig(os.path.join(path_dest, filename))


def main():
    print('Inference demo with CSL model')

    # Parsing args
    args = parse_args(parser)

    # Setup model
    print('Creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    # args.num_classes = state['num_classes']
    model = create_model(args).cuda()

    try:
        model.load_state_dict(state['model'], strict=True)
    except:
        model.load_state_dict(state, strict=True)

    model.eval()
    print('Done\n')

    # Convert class MID format to class description
    if args.dataset_type.lower() == 'flixstock':
        class_list = ['Neck 0', 'Neck 1', 'Neck 2', 'Neck 3', 'Neck 4', 'Neck 5', 'Neck 6', 
                        'Sleeve Length 0', 'Sleeve Length 1', 'Sleeve Length 2', 'Sleeve Length 3', 
                        'Pattern 0', 'Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4', 'Pattern 5', 
                        'Pattern 6', 'Pattern 7', 'Pattern 8', 'Pattern 9']
    else:
        class_list = np.array(list(state['idx_to_class'].values()))
        df_description = pd.read_csv(args.class_description_path)
        dict_desc = dict(zip(df_description.values[:, 0], df_description.values[:, 1]))
        class_list = [dict_desc[x] for x in class_list]

    # Inference
    print('Inference...\n')

    if args.dataset_type.lower() == 'flixstock' and args.all_val_inference:

        # TODO (get dataframe for test images and then loop the images for infering and save the result .csv)
        data_path_val   = os.path.join(args.data, 'images_split', 'val', 'images')    # args.data
        instances_path_val = os.path.join(args.data, 'attributes_val.csv')

        df = pd.read_csv(instances_path_val)

        neck_pred_list = []
        sleeve_length_pred_list = []
        pattern_pred_list = []

        for index in range(df.shape[0]):

            print('Inferring image ' + str(index+1) + ' / ' + str(df.shape[0]))

            image_name = df['filename'][index]

            image_path = os.path.join(data_path_val, image_name)
            tags, indexes, scores, im = inference(image_path, model, class_list, args)

            neck_pred_list.append(indexes[0])
            sleeve_length_pred_list.append(indexes[1])
            pattern_pred_list.append(indexes[2])

        df['Predicted Neck Type'] = neck_pred_list
        df['Predicted Sleeve Length Type'] = sleeve_length_pred_list
        df['Predicted Pattern Type'] = pattern_pred_list

        df.to_csv('./results/attributes_val_inference.csv', index=False)

    else:
        tags, indexes, scores, im = inference(args.pic_path, model, class_list, args)

    if not args.all_val_inference:
        # displaying image
        print('\nDisplay results...')
        display_image(im, tags, os.path.split(args.pic_path)[1])

        # # example loss calculation
        # output = model(tensor_batch)
        # loss_func1 = AsymmetricLoss()
        # loss_func2 = AsymmetricLossOptimized()
        # target = output.clone()
        # target[output < 0] = 0  # mockup target
        # target[output >= 0] = 1
        # loss1 = loss_func1(output, target)
        # loss2 = loss_func2(output, target)
        # assert abs((loss1.item() - loss2.item())) < 1e-6

        # displaying image
        print('showing image on screen...')
        fig = plt.figure()
        plt.imshow(im)
        plt.axis('off')
        plt.axis('tight')
        # plt.rcParams["axes.titlesize"] = 10
        plt.title("detected classes: {}".format(tags))

        print("\ndetected classes and scores: ")

        for c, s in zip(tags, scores):
            print(str(c) + ': ' + str(s))

        plt.show()

    print('done\n')



if __name__ == '__main__':
    main()
