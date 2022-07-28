import os
import json
# frames_root = '/home/cx/cx2/TrackingNet/TEST/frames'
# anno_root = '/home/cx/cx2/TrackingNet/TEST/anno'
# names = os.listdir(frames_root)
# TrackingNet = {}
# for name in names:
#     TrackingNet[name] = {}
#     TrackingNet[name]['video_dir'] = name
#     anno = os.path.join(anno_root,name+'.txt')
#     f = open(anno, 'r')
#     init_rect = f.readline()[0:-1].split(',')
#     init_rect = list(map(int, init_rect))
#     TrackingNet[name]['init_rect'] = init_rect
#     TrackingNet[name]['img_names'] = []
#     img_names = os.listdir(os.path.join(frames_root,name))
#     for i in range(len(img_names)):
#         TrackingNet[name]['img_names'].append(name + '/' + str(i) + '.jpg')
#     TrackingNet[name]['gt_rect'] = []
#     length = len(TrackingNet[name]['img_names'])
#     for i in range(length):
#         if i == 0:
#             TrackingNet[name]['gt_rect'].append(init_rect)
#         else:
#             TrackingNet[name]['gt_rect'].append([0, 0, 0, 0])
#
# TrackingNet = json.dumps(TrackingNet)
# with open('/home/cx/cx2/TrackingNet/TEST/TrackingNet.json', 'w') as json_file:
#     json_file.write(TrackingNet)


with open("/home/cx/cx2/Downloads/UAV123/UAV123_fix/Dataset_UAV123/UAV123/data_seq/UAV123/UAV.json", 'r') as f:
    temp = json.loads(f.read())
    b=1

