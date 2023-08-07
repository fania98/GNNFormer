"""
Extract cell graphs from the test set of the BRACS dataset.
"""

import os
from glob import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch 
from dgl.data.utils import save_graphs
from torchvision.models import resnet34

from histocartography.preprocessing import (
    VahadaneStainNormalizer,  # stain normalizer
    NucleiExtractor,          # nuclei detector 
    DeepFeatureExtractor,     # feature extractor 
    KNNGraphBuilder,          # graph builder,
    NucleiConceptExtractor    # concept extraction 
)

from histocartography.visualization import InstanceImageVisualization, OverlayGraphVisualization

# TUMOR_TYPE_TO_LABEL = {
#     'N': 0,
#     'PB': 0,
#     'UDH': 0,
#     'ADH': 1,
#     'FEA': 1,
#     'DCIS': 2,
#     'IC': 2
# }

MIN_NR_PIXELS = 500
MAX_NR_PIXELS = 50000000  

STAIN_NORM_TARGET_IMAGE = 'Report/Images/115832_007.png'  # define stain normalization target image.


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='path to the BRACS data.',
        default='Report/Images',
        required=False
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='path to save the cell graphs.',
        default='Report/cell_graph_k7',
        required=False
    )
    return parser.parse_args()


def generate_cell_graph(data_path, sample_file, save_path):
    """
    Generate a cell graph for all the images in image path dir.
    """

    # 1. get image path
    # subdirs = os.listdir(image_path)
    # image_fnames = []
    # for subdir in (subdirs + ['']):  # look for all the subdirs AND the image path
    #     image_fnames += glob(os.path.join(image_path, subdir, '*.png'))
    with open(os.path.join(sample_file)) as f:
        image_fnames = [os.path.join(data_path, line.strip()) for line in f.readlines()]

    print('*** Start analysing {} images ***'.format(len(image_fnames)))

    # 2. define stain normalizer 
    normalizer = VahadaneStainNormalizer(target_path=STAIN_NORM_TARGET_IMAGE)

    # 3. define nuclei extractor
    nuclei_detector = NucleiExtractor()
    viz = InstanceImageVisualization()

    # 4. define feature extractor: Extract patches of 72x72 pixels around each
    # nucleus centroid, then resize to 224 to match ResNet input size.
    feature_extractor = DeepFeatureExtractor(
        architecture='resnet34',
        patch_size=36,
        resize_size=224
    )

    # 5. define k-NN graph builder with k=5 and thresholding edges longer
    # than 50 pixels. Add image size-normalized centroids to the node features.
    # For e.g., resulting node features are 512 features from ResNet34 + 2
    # normalized centroid features.
    knn_graph_builder = KNNGraphBuilder(k=7, thresh=25, add_loc_feats=True)
    visualizer = OverlayGraphVisualization(node_style="fill", node_radius=0, node_color='wheat', edge_color="skyblue", edge_style=None, instance_visualizer=InstanceImageVisualization(instance_style="outline+filled"))
    # visualizer = InstanceImageVisualization(instance_style="outline+filled")

    # 6. define concept extractor (can take time on large images...)
    nuclei_concept_extractor = NucleiConceptExtractor()

    # 7. define var to store image IDs that failed (for whatever reason)
    image_ids_failing = []

    # 8. process all the images
    for image_path in tqdm(image_fnames):

        # a. load image & check if already there 
        _, image_name = os.path.split(image_path)
        image = np.array(Image.open(image_path))
        nr_pixels = image.shape[0] * image.shape[1]
        out_fname = os.path.join(save_path, image_name.replace('.png', '.bin'))

        # if file was not already created + not too big + not too small, then process 
        if not os.path.isfile(out_fname) and nr_pixels > MIN_NR_PIXELS and nr_pixels < MAX_NR_PIXELS:

            # b. stain norm the image
            # try:
            #     image = normalizer.process(image)
            #     print(image)
            # except:
            #     print('Warning: {} failed during stain normalization.'.format(image_path))
            #     image_ids_failing.append(image_path)
            #     pass

            # c. extract nuclei
            try:
                nuclei_map, _ = nuclei_detector.process(image)
                # img_viz = viz.process(canvas=image, graph=nuclei_map)
                # img_viz.save(os.path.join(args.save_path, image_name))
            except Exception as ex:
                print(ex)
                print('Warning: {} failed during nuclei detection.'.format(image_path))
                image_ids_failing.append(image_path)
                pass

            # d. extract deep features
            try:
                features = feature_extractor.process(image, nuclei_map)
            except:
                print('Warning: {} failed during deep feature extraction.'.format(image_path))
                image_ids_failing.append(image_path)
                pass

            # e. build a kNN graph
            try:
                graph = knn_graph_builder.process(nuclei_map, features)
                viz_cg = visualizer.process(canvas=image, graph=graph, instance_map=nuclei_map)
                viz_cg.save(os.path.join(args.save_path, image_name))
                print(f"save to {os.path.join(args.save_path, image_name)}")
            except:
                print('Warning: {} failed during kNN graph building.'.format(image_path))
                image_ids_failing.append(image_path)
                pass

            # f. extract nuclei-level concepts
            # try:
            #     concepts = nuclei_concept_extractor.process(image, nuclei_map)
            #     graph.ndata['concepts'] = torch.from_numpy(concepts).to(features.device)
            # except:
            #     print('Warning: {} failed during nuclei concept extraction.'.format(image_path))
            #     image_ids_failing.append(image_path)
            #     pass

            # g. save the graph

            # image_label = TUMOR_TYPE_TO_LABEL[image_name.split('_')[2]]
            save_graphs(
                filename=out_fname,
                g_list=[graph],
                # labels={"label": torch.tensor([image_label])}
            )

    image_ids_failing = list(set(image_ids_failing))
    print('Out of {} images, {} successful graph generations.'.format(
        len(image_fnames),
        len(image_fnames) - len(image_ids_failing)
    ))
    print('Failing IDs are:', image_ids_failing)

if __name__ == "__main__":

    # 1. handle i/o
    args = parse_arguments()
    if not os.path.isdir(args.data_path) or not os.listdir(args.data_path):
        raise ValueError("Data directory is either empty or does not exist.")
    os.makedirs(args.save_path, exist_ok=True)

    # 2. generate cell graphs one-by-one, will automatically
    # run on GPU if available. This process can take time depending 
    # on your hardware. Resulting output is expected to be 2GB. 
    # Running time is 1h15mn on NVIDIA GPU P100. 
    generate_cell_graph(
        data_path=args.data_path,
        sample_file= "Report/test.txt",
        save_path=args.save_path    
    )
