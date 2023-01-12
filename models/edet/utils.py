
from email.policy import default
import webcolors

BB_STANDARD_COLORS = ['LightCyan', 'AntiqueWhite', 'Beige', 'LightGrey', 'MistyRose', 'Bisque']

def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result

def standard_to_bgr(list_color_name, excluded=36):
    standard = []
    for i in range(len(list_color_name) - excluded):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index