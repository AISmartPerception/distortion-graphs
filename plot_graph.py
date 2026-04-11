import json
from graphviz import Digraph

color_palette = {
    "anchor": "#D9ACF5",
    "target": "#BBD8A3",
    "distortion_relation": "#BCCEF8",
    "distortion_attr": "#FFACC7",
    "scene_relation": "#FFD2A0",

    "anchor_border": "#80558C",
    "target_border": "#5F7161",
    "distortion_rel_border": "#093FB4",
    "distortion_attr_border": "#640D5F",
    "scene_rel_border": "#FE7743"
}

path = "inf_graphs/fig00_graph_fence.json"
output_name = path.split('/')[-1].split('.')[0]

with open(path, 'r') as f:
    data = json.load(f)

dot = Digraph(format='png')
# dot.attr(bgcolor='transparent') # transparent background if needed
dot.attr('node', fontsize='11', fontname='monospace')

image_1_objects = [obj for obj in data['objects'] if obj['image'] == str(1)]
image_2_objects = [obj for obj in data['objects'] if obj['image'] == str(2)]

for i, obj in enumerate(image_1_objects):
    dot.node(f'obj{i}_img1', obj['name'], 
             shape='rectangle',
             fillcolor=color_palette['anchor'],
             color=color_palette['anchor_border'],
             penwidth="1.5",
             style='filled,rounded',
             width='0.01',
             height='0.1')

for i, obj in enumerate(image_2_objects):
    dot.node(f'obj{i}_img2', obj['name'], 
             shape='rectangle',
             fillcolor=color_palette['target'],
             color=color_palette['target_border'],
             penwidth="1.5",
             style='filled,rounded',
             width='0.01',
             height='0.1')
    
attributes = data['attributes']
seen_attributes = set()
for attribute in attributes:
    attr_name = attribute['attribute']
    belongs_to = int(attribute['object'])
    img_id = int(attribute['image'])
    if img_id == 2:
        belongs_to -= len(image_1_objects)
    
    # avoid duplicates if they are added in the previous
    if (attr_name, belongs_to, img_id) in seen_attributes:
        continue

    dot.node(f'{attr_name}_{belongs_to}_{img_id}', attr_name, 
             shape='rectangle', 
             fillcolor=color_palette['distortion_attr'],
             color=color_palette['distortion_attr_border'],
             penwidth="1.5",
             style='filled,rounded',
             width='0.01',
             height='0.1')
    dot.edge(f'obj{belongs_to}_img{img_id}', f'{attr_name}_{belongs_to}_{img_id}')
    seen_attributes.add((attr_name, belongs_to, img_id))

# create relationships for each image (internal relationships within same image)
# relationships = data['relationships']
# seen_relationships = set()
# for rel in relationships:
#     rel_name = rel['predicate']
#     subject_id = int(rel['subject'])
#     object_id = int(rel['object'])
    
#     print(rel)
#     same_subject = [x for x in data['objects'] if int(x['id']) == subject_id][0]
#     same_object = [x for x in data['objects'] if int(x['id']) == object_id][0]
#     subject_image = int(same_subject['image'])
#     object_image = int(same_object['image'])
#     img_id = rel['image']

#     if img_id == 2:
#         object_id -= len(image_1_objects)
#         subject_id -= len(image_1_objects)

#     # Create relationship nodes only if subject and object belong to the same image
#     if subject_image == object_image:
#         subject_prefix = f"obj{subject_id}_img{subject_image}"
#         object_prefix = f"obj{object_id}_img{object_image}"
#         # avoid duplicates if they are added in the previous
#         if (subject_id, object_id, subject_image, object_image) in seen_relationships:
#             continue

#         dot.node(f'rel{subject_id}_{object_id}_{subject_image}_{object_image}', rel_name, 
#                  shape='rectangle', 
#                  fillcolor=color_palette['scene_relation'],
#                  color=color_palette['scene_rel_border'],
#                  penwidth="1.5",
#                  style='filled,rounded',
#                  width='0.01',
#                  height='0.1')
        
#         dot.edge(f'{subject_prefix}', f'rel{subject_id}_{object_id}_{subject_image}_{object_image}')
#         dot.edge(f'rel{subject_id}_{object_id}_{subject_image}_{object_image}', f'{object_prefix}')
#         seen_relationships.add((subject_id, object_id, subject_image, object_image))

# Create cross-image comparison relationships for same objects
across_relationships = data['art']
for rel in across_relationships:
    rel_name = rel['predicate']
    subject_id = int(rel['subject'])
    object_id = int(rel['object'])
    same_subject = [x for x in data['objects'] if int(x['id']) == subject_id][0]
    same_object = [x for x in data['objects'] if int(x['id']) == object_id][0]

    subject_image = int(same_subject['image'])
    object_image = int(same_object['image'])
    
    if subject_image != object_image and same_subject['name'] == same_object['name']:

        if subject_image == 2:
            subject_id -= len(image_1_objects)
        if object_image == 2:
            object_id -= len(image_1_objects)

        subject_prefix = f"obj{subject_id}_img1"
        object_prefix = f"obj{object_id}_img2"
        
        dot.node(f'rel{subject_id}_{object_id}_comp', rel_name, 
                 shape='rectangle',
                 fillcolor=color_palette['distortion_relation'],
                 color=color_palette['distortion_rel_border'],
                 penwidth="1.5",
                 style='filled,rounded',
                 width='0.01',
                 height='0.1')
        
        dot.edge(f'{subject_prefix}', f'rel{subject_id}_{object_id}_comp')
        dot.edge(f'rel{subject_id}_{object_id}_comp', f'{object_prefix}')

dot.attr(dpi='300')
dot.render(f'graphs/{output_name}', format='png', view=False)