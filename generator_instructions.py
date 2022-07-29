import glob
PATH = "/root/share/gym/gym/envs/mujoco/assets/sim_push_xmls"
files = glob.glob(PATH+'/*.xml')
# print(len(files))
# for file in files[:10]:
#     print(file)

import re
import collections
import xml.etree.ElementTree as ET
object_dicts = []
for file in files:
    d = {}
    d["file"] = file
    tree = ET.parse(file)
    root = tree.getroot()
    for child in root:
#         print(child.tag)
        if child.tag == "asset":
            for i in child:
#                 print(i)
                if i.tag == "mesh":
#                     print(i.attrib["name"], i.attrib["file"])
                    mesh = i.attrib["file"]
                    mesh = re.sub("\d+", "", mesh)
                    mesh = mesh.split("/")[-1]
                    mesh = mesh.split(".")[0]
                    mesh = mesh.replace("_", " ")
                    mesh = mesh.replace("-", " ")   
                    d[i.attrib["name"]] = mesh
                if i.tag == "texture":
#                     print(i.attrib["name"], i.attrib["file"])
                    texture = i.attrib["file"]  #  textures/obj_textures/pleated_0105.png
                    texture = re.sub("\d+", "", texture)
                    texture = texture.split("/")[-1]  # 'pleated_0105.png'
                    texture = texture.split(".")[0]  # 'pleated_0105'
                    texture = texture.split("_")[0]  # 'pleated'
                    texture = texture.replace("-", " ")
                    d[i.attrib["name"]] = texture
        
    adj = d['object']
    n = d['object_mesh']
#     d["instruction"] = f"Push the {adj} {n} to the red area."
    d["instruction"] = []
    d["instruction"].append(f"Push the {n} to the red area.")
    d["instruction"].append(f"Push the {n} to the red part.")
    d["instruction"].append(f"Push the {n} to the goal.")
    d["instruction"].append(f"Let the {n} be in the red area.")
    d["instruction"].append(f"Let the {n} be in the red part.")
    d["instruction"].append(f"Let the {n} be in the goal.")
    d["instruction"].append(f"Have the {n} in the red area.")
    d["instruction"].append(f"Have the {n} in the red part.")
    d["instruction"].append(f"Have the {n} in the goal.")
    d["instruction"].append(f"Make the {n} in the red area.")
    d["instruction"].append(f"Make the {n} in the red part.")
    d["instruction"].append(f"Make the {n} in the goal.")
    d["instruction"].append(f"Shobe the {n} to the red area.")
    d["instruction"].append(f"Shobe the {n} to the red part.")
    d["instruction"].append(f"Shobe the {n} to the goal.")
    d["word_instruction"] = [f'{n}']
    assert len(d) == 8
    object_dicts.append(d)
object_dicts.sort(key=lambda x:x["file"])
print(len(object_dicts))
print(object_dicts[0])# {'file': '/root/workspace/gym/gym/envs/mujoco/assets/sim_push_xmls/test2_ensure_woodtable_distractor_pusher0.xml', 'table': 'wpic', 'object_mesh': 'Recycle Soda Can', 'distractor_mesh': 'Elephant', 'distractor': 'zigzagged', 'object': 'studded'}
# print(objects)  # ['ball holder', 'pleated'] [ n. , adj.]
# print(distractors)  # ['Simple Filament Guide', 'polka dotted']


from bert_serving.client import BertClient
bc = BertClient()
a = []
import numpy as np
import os
from tqdm import tqdm 
for i in tqdm(range(len(object_dicts))):
# for i in tqdm(range(5)):
#     print(i)
    file_name = object_dicts[i]["file"]
    file_name = file_name.split("/")[-1][:-4]
    instruction_encoding = bc.encode(object_dicts[i]["instruction"])
#     instruction_encoding = instruction_encoding.reshape([1, 768])
#     print(np.array(instruction_encoding).shape)
    a.append(instruction_encoding)
    path = '/root/share/768_15_types_instruction/'
#     print(path + file_name)
    os.makedirs(path, exist_ok=True)
    np.save(path + file_name, instruction_encoding)