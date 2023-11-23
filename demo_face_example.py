#
from pprint import pprint
from applications.mixed_reality import run_gangealing_on_video, run_gangealing_on_webcam
from applications import load_stn
from utils.download import download_model, download_video


model = '🧑🏼‍🔬 human (celeba)'  # @param ['🧑🏼‍🔬 human (celeba)', '🐶 dog', '🐱 cat', '🐦 bird (cub)', '🚲 bicycle', '📺 tvmonitor']
video = '🧑🏼‍🔬 elon'  # @param ["🐱 cutecat", "🧑🏼‍🔬 elon", "🐶 snowpuppy", "Upload my own video"]

video_download_mode = 'fast'  #@param ['fast', 'high quality']
video_online_dir = 'video_1024' if video_download_mode == 'high quality' else 'video'
model = model.split(' ')
if len(model) == 2:
  model = model[1]
else:
  model = model[2][1:-1]
checkpoint = download_model(model, 'pretrained_stn_only')

video = video[video.rfind(" ")+1:]
if video != 'Upload my own video':
  video_path = download_video(video, video_online_dir)
  video_size = 1024 if video_download_mode == 'high quality' else 512

video_path = "data/video_frames/test_1024/test_1080.mp4"
video_size = 512

fps = 30  #@param {type:"integer"}
# blend_alg = 'alpha'  #@param ["alpha", "laplacian", "laplacian_light"]
blend_alg = 'laplacian'  #@param ["alpha", "laplacian", "laplacian_light"]
batch_size = 1  #@param {type:"integer"}
use_flipping = False  #@param {type:"boolean"}
memory_efficient_but_slower = False  #@param {type:"boolean"}


class MyDict():
  def __init__(self): pass

# Assign a bunch of arguments. For some reason, this demo
# runs way faster when invoking python commands directly 
# than calling python from bash.
args = MyDict()
args.real_size = int(video_size)
args.real_data_path = video_path
args.fps = fps
args.batch = batch_size
args.blend_alg = blend_alg
args.transform = ['similarity', 'flow']
args.flow_size = 128
args.stn_channel_multiplier = 0.5
args.num_heads = 1
args.distributed = False  # Colab only uses 1 GPU
args.clustering = False
args.cluster = None
args.objects = True
args.no_flip_inference = not use_flipping
args.save_frames = memory_efficient_but_slower
args.overlay_congealed = False
args.ckpt = model
args.override = False
args.save_correspondences = False
args.out = 'visuals'
object_picker_value = 'dense tracking'
if object_picker_value == 'dense tracking':
  args.label_path = f'assets/masks/{model}_mask.png'
  # Feel free to change the parameters below:
  args.resolution = 128
  args.sigma = 1.3
  args.opacity = 0.8
  args.objects = False
else:  # object lense
  args.label_path = f'assets/objects/{model}/{object_picker_value}'
  args.resolution = 4 * int(video_size)
  args.sigma = 0.8
  args.opacity = 0.8
  # args.objects = True

# args.label_path = "assets/objects/celeba/celeba_moustache_1024.png"
args.sigma = 0.8
args.opacity = 0.8
args.DEBUG = True
pprint(args.__dict__)
# args.label_path = "assets/objects/celeba/celeba_moustache.png"
# args.label_path = "assets/objects/celeba/celeba_pokemon.png"
args.label_path = "assets/objects/horse_cluster2/horse_cluster2_unicorn.png"

# Run Mixed Reality!
stn = load_stn(args, device="cuda")
print('Running Spatial Transformer on frames...')

# run_gangealing_on_video(args, stn, classifier=None)
run_gangealing_on_webcam(args, stn, classifier=None)
