import numpy as np
import cv2
import argparse

import os
import torch

from pathlib import Path
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


def ranking_score(matches, match_confidence):
    return np.sum(np.multiply(matches,match_confidence)).astype(np.float32)


def plot_matches(args, match_map):
    for query, result_set in match_map.items():
        reference, percentage = result_set
        if reference is None or percentage is None:
            continue
        query_filename = os.path.join(args.query_dir, f"{query}.jpg")
        ref_filename = os.path.join(args.input_dir, f"{reference}.jpg")
        query_image = cv2.cvtColor(cv2.imread(query_filename), cv2.COLOR_BGR2RGB)
        ref_image = cv2.cvtColor(cv2.imread(ref_filename), cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(query_image)
        axes[0].set_title(f"Q: {query}")
        axes[0].axis('off')

        axes[1].imshow(ref_image)
        axes[1].set_title(f"R: {reference}, Match(%): {percentage}")
        axes[1].axis('off')
        output_file_name = os.path.join(args.output_dir, f"{query}_{reference}.jpg")
        plt.savefig(output_file_name)



class theMatcher:
    """
    A solution class for query-reference image matching using SuperGlue. An instance of this class need only
    be created once using the necessary argparse parameters. Then the method can be repeatedly invoked with
    a new query image. 

    """

    def __init__(self, input_args):
        self.args = input_args

    
    def run_query_ref_matching(self, query):
        """
        A function to perform matching for all query images in the provided query directory against
        reference images in the input directory. 

        Returns a dictionary mapping the match percentage of every reference image against the query.
        """

        # score for each image to query image
        score_dict = {}

        if len(self.args.resize) == 2 and self.args.resize[1] == -1:
            self.args.resize = self.args.resize[0:1]
        if len(self.args.resize) == 2:
            print('Will resize to {}x{} (WxH)'.format(
                self.args.resize[0], self.args.resize[1]))
        elif len(self.args.resize) == 1 and self.args.resize[0] > 0:
            print('Will resize max dimension to {}'.format(self.args.resize[0]))
        elif len(self.args.resize) == 1:
            print('Will not resize images')
        else:
            raise ValueError('Cannot specify more than two integers for --resize')

        all_reference_image_names = os.listdir(self.args.input_dir)

        with open('rank_pairs.txt', 'w') as file:
            # first line should map query to query for basis matching
            file.write(f'{query} {query}\n')
            # continue adding query-reference pairs
            for reference_name in all_reference_image_names:
                if (reference_name.endswith('.jpg') or reference_name.endswith('.png')):
                    file.write(f'{query} {reference_name}\n')

        with open('rank_pairs.txt', 'r') as f:
            pairs = [l.split() for l in f.readlines()]

        if self.args.max_length > -1:
            pairs = pairs[0:np.min([len(pairs), self.args.max_length])]

        # Load the SuperPoint and SuperGlue models.
        device = 'cuda' if torch.cuda.is_available() and not self.args.force_cpu else 'cpu'
        print('Running inference on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': self.args.nms_radius,
                'keypoint_threshold': self.args.keypoint_threshold,
                'max_keypoints': self.args.max_keypoints
            },
            'superglue': {
                'weights': self.args.superglue,
                'sinkhorn_iterations': self.args.sinkhorn_iterations,
                'match_threshold': self.args.match_threshold,
            }
        }

        matching = Matching(config).eval().to(device)

        # Create the output directories if they do not exist already.
        input_dir = Path(self.args.input_dir)
        print('Looking for data in directory \"{}\"'.format(input_dir))

        query_dir = Path(self.args.query_dir)
        print('Looking for query in directory \"{}\"'.format(query_dir))
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        print('Will write matches to directory \"{}\"'.format(output_dir))

        if self.args.viz:
            print('Will write visualization images to',
                  'directory \"{}\"'.format(output_dir))

        timer = AverageTimer(newline=True)
        for i, pair in enumerate(pairs):
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
            eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
            viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, self.args.viz_extension)
            viz_eval_path = output_dir / \
                '{}_{}_evaluation.{}'.format(stem0, stem1, self.args.viz_extension)

            do_match = True
            do_viz = self.args.viz
            if self.args.viz and viz_path.exists():
                do_viz = False

            if not (do_match or do_viz):
                timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
                continue

            # If a rotation integer is provided (e.g. from EXIF data), use it:
            if len(pair) >= 5:
                rot0, rot1 = int(pair[2]), int(pair[3])
            else:
                rot0, rot1 = 0, 0

            # Load the image pair.
            if name0 == name1:
                image0, inp0, scales0 = read_image(
                    query_dir / name0, device, self.args.resize, rot0, self.args.resize_float)
                image1, inp1, scales1 = read_image(
                    query_dir / name1, device, self.args.resize, rot1, self.args.resize_float)
            else:
                image0, inp0, scales0 = read_image(
                    query_dir / name0, device, self.args.resize, rot0, self.args.resize_float)
                image1, inp1, scales1 = read_image(
                    input_dir / name1, device, self.args.resize, rot1, self.args.resize_float)
                if image0 is None or image1 is None:
                    print('Problem reading image pair: {} {}'.format(
                        query_dir/name0, input_dir/name1))
                    exit(1)
            timer.update('load_image')

            if do_match:
                # Perform the matching.
                pred = matching({'image0': inp0, 'image1': inp1})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']
                timer.update('matcher')

                # Write the matches to disk.
                out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                               'matches': matches, 'match_confidence': conf}

                # print('[DEBUGGING] matches:', matches)
                # print('[DEBUGGING] matches shape:', matches.shape)
                # print('[DEBUGGING] conf:', conf)
                # print('[DEBUGGING] conf shape:', conf.shape)

                # save score to score dict
                score_dict[stem1] = ranking_score(matches, conf)

                # save full score to calculate %
                if name0 == name1:
                    full_score = score_dict[stem1]

                # save to .npz file
                # np.savez(str(matches_path), **out_matches)

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            if do_viz:
                # Visualize the matches.
                color = cm.jet(mconf)
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0)),
                ]
                if rot0 != 0 or rot1 != 0:
                    text.append('Rotation: {}:{}'.format(rot0, rot1))

                # Display extra parameter info.
                k_thresh = matching.superpoint.config['keypoint_threshold']
                m_thresh = matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                    'Image Pair: {}:{}'.format(stem0, stem1),
                ]

                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path, self.args.show_keypoints,
                    self.args.fast_viz, self.args.opencv_display, 'Matches', small_text)

                timer.update('viz_match')

            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

        ranked_images = {k:v for k,v in sorted(score_dict.items(), reverse = True, key= lambda x: x[1])}

        ranked_images_percentage = {k:f'{((v/full_score)*100):.3f}' for k,v in ranked_images.items()}

        ####write ranked image .csv
        df = pd.DataFrame.from_dict(ranked_images_percentage,orient='index',columns = ['score'])
        df.reset_index(inplace=True)
        df.rename(columns = {'index':'image'},inplace=True)
        df.to_csv(str(output_dir/'ranking_score.csv'), index=True)
        print(ranked_images_percentage)

        return ranked_images_percentage

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir', type=str, required=True,
        help='Path to database image directory')
    parser.add_argument(
        '-s', '--query_dir', type=str, required=True,
        help='Path to query image directory')
    parser.add_argument(
        '-o', '--output_dir', type=str, default='rank_output/',
        help='Path to store npz and visualization files')
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[160, 120],
        help='Resize the input image before running inference. If two numbers, '
        'resize to the exact dimensions, if one number, resize the max '
        'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
        ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')

    args = parser.parse_args()

    assert not (args.opencv_display and not args.viz), 'Must use --viz with --opencv_display'
    assert not (args.opencv_display and not args.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (args.fast_viz and not args.viz), 'Must use --viz with --fast_viz'
    assert not (args.fast_viz and args.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'



    # traverse the query directory and create a list of query images
    query_list = []
    for root, dirs, files in os.walk(args.query_dir):
        for file in files:
            query_list.append(file)
    
    match_map = {}
    matcher = theMatcher(args)
    for query in query_list:
        image_rank_dict = matcher.run_query_ref_matching(query)
        print(f"Matching for {query}: \n")
        print(f"{image_rank_dict}\n")
        query_name = query.split('.')[0]
        while True:
            match, percentage = max(image_rank_dict.items(), key=lambda item: float(item[1]))
            if match == query_name:
                del image_rank_dict[match]
            else:
                break
        if float(percentage) <= 10.0:
            match_map[query_name] = (None, None)
        else:
            match_map[query_name] = (match, percentage)

    print(f"Matches found: {match_map}")
    plot_matches(args, match_map)

