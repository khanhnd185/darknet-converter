import abc
import collections
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def _get_multiplier_for_color_randomness():
  """Returns a multiplier to get semi-random colors from successive indices.

  This function computes a prime number, p, in the range [2, 17] that:
  - is closest to len(STANDARD_COLORS) / 10
  - does not divide len(STANDARD_COLORS)

  If no prime numbers in that range satisfy the constraints, p is returned as 1.

  Once p is established, it can be used as a multiplier to select
  non-consecutive colors from STANDARD_COLORS:
  colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
  """
  num_colors = len(STANDARD_COLORS)
  prime_candidates = [5, 7, 11, 13, 17]

  # Remove all prime candidates that divide the number of colors.
  prime_candidates = [p for p in prime_candidates if num_colors % p]
  if not prime_candidates:
    return 1

  # Return the closest prime number to num_colors / 10.
  abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
  num_candidates = len(abs_distance)
  inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
  return prime_candidates[inds[0]]


def _limit_boxcoord(x, xmin, xmax):
  return min(max(x,xmin), xmax)


def draw_bounding_box_on_image(
  image, box,
  box_format="x0y0x1y1wh",
  color='red',
  thickness=4,
  fontsize=10,
  display_str_list=(),
  use_normalized_coordinates=True
):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box (each to be shown on its
      own line).
    use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
      ymax, xmax as relative to the image.  Otherwise treat coordinates as
      absolute.
  """
  draw  = ImageDraw.Draw(image)
  IW,IH = image.size
  if   (box_format=="x0y0wh"):
    [xmin,ymin,xmax,ymax] = [box[0],         box[1],         box[0]+box[2],  box[1]+box[3]  ]
  elif (box_format=="xcycwh"):
    [xmin,ymin,xmax,ymax] = [box[0]-box[2]/2,box[1]-box[3]/2,box[0]+box[2]/2,box[1]+box[3]/2]
  elif (box_format in ["x0y0x1y1","x0y0x1y1wh"]):
    [xmin,ymin,xmax,ymax] =  box[:4]
  elif (box_format=="y0x0y1x1"):
    [ymin,xmin,ymax,xmax] =  box
  else: raise ValueError(f"Unknown box_format=\"{box_format}\" provided.")

  if use_normalized_coordinates:
    ymin  = _limit_boxcoord(ymin*IH, 0, IH          )
    xmin  = _limit_boxcoord(xmin*IW, 0, IW          )
    ymax  = _limit_boxcoord(ymax*IH, 0, IH-thickness)
    xmax  = _limit_boxcoord(xmax*IW, 0, IW-thickness)
  else:
    ymin  = _limit_boxcoord(ymin   , 0, IH          )
    xmin  = _limit_boxcoord(xmin   , 0, IW          )
    ymax  = _limit_boxcoord(ymax   , 0, IH-thickness)
    xmax  = _limit_boxcoord(xmax   , 0, IW-thickness)

  if (thickness>0):
    draw.line(
      [(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)],
      width=thickness, fill=color
    )

  try:            font = ImageFont.truetype('arial.ttf', fontsize)
  except IOError: font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getbbox(_)[3]-font.getbbox(_)[1] for _ in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2*0.05) * sum(display_str_heights)

  if (ymin>total_display_str_height):
        text_bottom = ymin
  else: text_bottom = ymax+total_display_str_height

  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    x0,y0,x1,y1 = font.getbbox(display_str)
    x0         += xmin
    x1         += xmin
    text_width, text_height = (x1-x0, y1-y0)
    margin                  = np.ceil(0.05*text_height)
    draw.rectangle([(x0, text_bottom-text_height-2*margin),
                    (x0+text_width, text_bottom)],
                   fill=color)
    draw.text((x0+margin, text_bottom-text_height-margin), display_str,
              fill='black', font=font)
    text_bottom -= text_height-2*margin


def draw_keypoints_on_image(
  image,
  keypoints,
  color='red',
  radius=2,
  use_normalized_coordinates=True,
  keypoint_edges=None,
  keypoint_edge_color='green',
  keypoint_edge_width=2
):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    keypoint_edge_color: color to draw the keypoint edges with. Default is red.
    keypoint_edge_width: width of the edges drawn between keypoints. Default
      value is 2.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color,
                 fill=color)
  if keypoint_edges is not None:
    for keypoint_start, keypoint_end in keypoint_edges:
      if (keypoint_start < 0 or keypoint_start >= len(keypoints) or
          keypoint_end < 0 or keypoint_end >= len(keypoints)):
        continue
      edge_coordinates = [
          keypoints_x[keypoint_start], keypoints_y[keypoint_start],
          keypoints_x[keypoint_end], keypoints_y[keypoint_end]
      ]
      draw.line(
          edge_coordinates, fill=keypoint_edge_color, width=keypoint_edge_width)


def draw_mask_on_image(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: a PIL.Image object
    mask: a uint8 numpy array of shape (img_height, img_width) with values
      between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if (image.size[0]!=mask.shape[1] or image.size[1]!=mask.shape[0]):
    raise ValueError("The image size %s is not matched with the mask shape %s"
      %(image.size, mask.shape))
  rgb = ImageColor.getrgb(color)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
  image = Image.composite(pil_solid_color, image, pil_mask)


def visualize_image(
    image,
    scores,
    classes,
    boxes,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_edges=None,
    track_ids=None,
    box_format="x0y0x1y1wh",
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    agnostic_mode=False,
    line_thickness=2,
    label_fontsize=10,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False
  ):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then this
      function assumes that the boxes to be plotted are groundtruth boxes and
      plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can be None
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    box_format: box coordinates format
    use_normalized_coordinates: whether boxes is to be interpreted as normalized
      coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw all
      boxes.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_boxes: whether to skip the drawing of bounding boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break

    box = tuple(boxes[i].tolist())
    if instance_masks is not None:
      box_to_instance_masks_map[box] = instance_masks[i]
    if instance_boundaries is not None:
      box_to_instance_boundaries_map[box] = instance_boundaries[i]
    if keypoints is not None:
      box_to_keypoints_map[box].extend(keypoints[i])
    if track_ids is not None:
      box_to_track_ids_map[box] = track_ids[i]
    if scores is None:
      box_to_color_map[box] = groundtruth_box_visualization_color
    else:
      display_str = ''
      if not skip_labels:
        if not agnostic_mode:
          if classes[i] in six.viewkeys(category_index):
            class_name = category_index[classes[i]]['name']
          else:
            class_name = 'N/A'
          display_str = str(class_name)
      if not skip_scores:
        if not display_str:
          display_str = '{}%'.format(int(100 * scores[i]))
        else:
          display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
      if not skip_track_ids and track_ids is not None:
        if not display_str:
          display_str = 'ID {}'.format(track_ids[i])
        else:
          display_str = '{}: ID {}'.format(display_str, track_ids[i])
      box_to_display_str_map[box].append(display_str)
      if agnostic_mode:
        box_to_color_map[box] = 'DarkOrange'
      elif track_ids is not None:
        prime_multipler = _get_multiplier_for_color_randomness()
        box_to_color_map[box] = STANDARD_COLORS[(prime_multipler *
                                                 track_ids[i]) %
                                                len(STANDARD_COLORS)]
      else:
        box_to_color_map[box] = STANDARD_COLORS[classes[i] %
                                                len(STANDARD_COLORS)]

  ##  Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    if (instance_masks):
      draw_mask_on_image(
        image, box_to_instance_masks_map[box],
        color=color
      )
    if (instance_boundaries):
      draw_mask_on_image(
        image, box_to_instance_boundaries_map[box],
        color='red', alpha=1.0
      )
    draw_bounding_box_on_image(
      image, box,
      box_format=box_format,
      color=color,
      thickness=0 if skip_boxes else line_thickness,
      fontsize=label_fontsize,
      display_str_list=box_to_display_str_map[box],
      use_normalized_coordinates=use_normalized_coordinates
    )
    if (keypoints):
      draw_keypoints_on_image(
       image, box_to_keypoints_map[box],
       color=color,
       radius=line_thickness / 2,
       use_normalized_coordinates=use_normalized_coordinates,
       keypoint_edges=keypoint_edges,
       keypoint_edge_color=color,
       keypoint_edge_width=line_thickness // 2
     )
  return  image
