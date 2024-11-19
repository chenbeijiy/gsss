import os
from argparse import ArgumentParser

dtu_scenes =  ['scan122']
# dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/dtu")
parser.add_argument('--dtu', "-dtu", default="../data/dtu-2dgs")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", default="../data/dtu-2dgs")
    # parser.add_argument('--eval_path', required=True, type=str)
    args = parser.parse_args()


if not args.skip_training:
    common_args = " --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000"
    for scene in dtu_scenes:
        source = "../data/dtu-2dgs" + "/" + scene
        depth_path = "./depth/dtu/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + "/" + " -d " + depth_path + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    common_args = " --quiet --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    for scene in dtu_scenes:
        source = "../data/dtu-2dgs" + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)


# metrics.py
# for scene in dtu_scenes:
#     tar = args.output_path + "/" + scene
#     os.system("python metrics.py -m"+tar)


if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for scene in dtu_scenes:
        # print("metrics "+dtu_scenes)
        scan_id = scene[4:]
        ply_file = f"{args.output_path}/{scene}/train/ours_30000/"
        iteration = 30000
        string = f"python ./scripts/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {args.output_path}/{scene}/train/ours_30000/fuse_post.ply " + \
            f"--scan_id {scan_id} --output_dir {args.output_path}/scan{scan_id}/tmp " + \
            f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
        
        os.system(string)
