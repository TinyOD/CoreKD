import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from ..utils import mask2ndarray

EPS = 1e-2

def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_det_bboxes(
                      img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'


    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)


    # bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    bbox_color_ap = color_val_matplotlib((48, 129, 244))
    bbox_color_at = color_val_matplotlib((159, 68, 255))
    bbox_color_basef = color_val_matplotlib((130, 74, 192))
    bbox_color_basketc = color_val_matplotlib((215, 223, 5))
    bbox_color_br = color_val_matplotlib((255, 224, 24))
    bbox_color_chi = color_val_matplotlib((238, 238, 238))
    bbox_color_dam = color_val_matplotlib((144, 216, 85))
    bbox_color_esa = color_val_matplotlib((246, 51, 92))
    bbox_color_ets = color_val_matplotlib((66, 255, 240))
    bbox_color_golf = color_val_matplotlib((207, 170, 255))
    bbox_color_gtf = color_val_matplotlib((123, 166, 223))
    bbox_color_har = color_val_matplotlib((129, 219, 139))
    bbox_color_over = color_val_matplotlib((0, 0, 99))

    # bbox_color_sh = color_val_matplotlib((239, 144, 198))
    bbox_color_sh = color_val_matplotlib((0, 255, 0)) # SSDD

    bbox_color_sta = color_val_matplotlib((202, 255, 0))
    bbox_color_st = color_val_matplotlib((75, 24, 229))
    bbox_color_tenn = color_val_matplotlib((111, 29, 22))
    bbox_color_train = color_val_matplotlib((125, 110, 8))
    bbox_color_veh = color_val_matplotlib((0, 255, 0))
    bbox_color_wm = color_val_matplotlib((24, 224, 255))

    bbox_color_pe = color_val_matplotlib((33, 0, 98))
    bbox_color_sw = color_val_matplotlib((173, 228, 219))

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons_ap = []
    color_ap = []

    polygons_basketc = []
    color_basketc = []

    polygons_basef = []
    color_basef = []

    polygons_chi = []
    color_chi = []

    polygons_dam = []
    color_dam = []

    polygons_esa = []
    color_esa = []

    polygons_ets = []
    color_ets = []

    polygons_golf = []
    color_golf = []

    polygons_gtf = []
    color_gtf = []

    polygons_har = []
    color_har = []

    polygons_over = []
    color_over = []

    polygons_sta = []
    color_sta = []

    polygons_tenn = []
    color_tenn = []

    polygons_train = []
    color_train = []


    polygons_at = []
    color_at = []

    polygons_sh = []
    color_sh = []

    polygons_pe = []
    color_pe = []

    polygons_br = []
    color_br = []

    polygons_sw = []
    color_sw = []

    polygons_wm = []
    color_wm = []

    polygons_st = []
    color_st = []

    polygons_veh = []
    color_veh = []

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'

        if 'airplane' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_ap.append(Polygon(np_poly))
            color_ap.append(bbox_color_ap)

        if 'airport' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_at.append(Polygon(np_poly))
            color_at.append(bbox_color_at)

        if 'baseballfield' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_basef.append(Polygon(np_poly))
            color_basef.append(bbox_color_basef)

        if 'basketballcourt' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_basketc.append(Polygon(np_poly))
            color_basketc.append(bbox_color_basketc)

        if 'bridge' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_br.append(Polygon(np_poly))
            color_br.append(bbox_color_br)
            # print('img_bridge:', img_dir)

        if 'chimney' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_chi.append(Polygon(np_poly))
            color_chi.append(bbox_color_chi)

        if 'dam' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_dam.append(Polygon(np_poly))
            color_dam.append(bbox_color_dam)


        if 'Expressway-Service-area' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_esa.append(Polygon(np_poly))
            color_esa.append(bbox_color_esa)

        if 'Expressway-toll-station' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_ets.append(Polygon(np_poly))
            color_ets.append(bbox_color_ets)
            # print('img_ets:', img_dir)

        if 'golffield' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_golf.append(Polygon(np_poly))
            color_golf.append(bbox_color_golf)

        if 'groundtrackfield' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_gtf.append(Polygon(np_poly))
            color_gtf.append(bbox_color_gtf)

        if 'harbor' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_har.append(Polygon(np_poly))
            color_har.append(bbox_color_har)
            # print('img_harbor:', img_dir)


        if 'overpass' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_over.append(Polygon(np_poly))
            color_over.append(bbox_color_over)
            # print('img_overpass:', img_dir)

        if 'stadium' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_sta.append(Polygon(np_poly))
            color_sta.append(bbox_color_sta)


        if 'tenniscourt' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_tenn.append(Polygon(np_poly))
            color_tenn.append(bbox_color_tenn)

        if 'trainstation' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_train.append(Polygon(np_poly))
            color_train.append(bbox_color_train)

        if 'storagetank' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_st.append(Polygon(np_poly))
            color_st.append(bbox_color_st)

        if 'ship' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_sh.append(Polygon(np_poly))
            color_sh.append(bbox_color_sh)

        if 'swimming-pool' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_sw.append(Polygon(np_poly))
            color_sw.append(bbox_color_sw)

        if 'vehicle' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_veh.append(Polygon(np_poly))
            color_veh.append(bbox_color_veh)

        if 'person' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_pe.append(Polygon(np_poly))
            color_pe.append(bbox_color_pe)

        if 'windmill' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons_wm.append(Polygon(np_poly))
            color_wm.append(bbox_color_wm)

        # ax.text(
        #     bbox_int[0],
        #     bbox_int[1],
        #     f'{label_text}',
        #     bbox={
        #         'facecolor': 'none',
        #         'alpha': 0,
        #         'pad': 0,
        #         'edgecolor': 'none'
        #     },
        #     color=text_color,
        #     fontsize=font_size,
        #     verticalalignment='top',
        #     horizontalalignment='left')
        # print('label_text:', label_text)

        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p_ap = PatchCollection(
        polygons_ap, facecolor='none', edgecolors=color_ap, linewidths=thickness)
    ax.add_collection(p_ap)

    p_at = PatchCollection(
        polygons_at, facecolor='none', edgecolors=color_at, linewidths=thickness)
    ax.add_collection(p_at)

    p_basef = PatchCollection(
        polygons_basef, facecolor='none', edgecolors=color_basef, linewidths=thickness)
    ax.add_collection(p_basef)

    p_basketc = PatchCollection(
        polygons_basketc, facecolor='none', edgecolors=color_basketc, linewidths=thickness)
    ax.add_collection(p_basketc)

    p_chi = PatchCollection(
        polygons_chi, facecolor='none', edgecolors=color_chi, linewidths=thickness)
    ax.add_collection(p_chi)

    p_dam = PatchCollection(
        polygons_dam, facecolor='none', edgecolors=color_dam, linewidths=thickness)
    ax.add_collection(p_dam)

    p_esa = PatchCollection(
        polygons_esa, facecolor='none', edgecolors=color_esa, linewidths=thickness)
    ax.add_collection(p_esa)

    p_ets = PatchCollection(
        polygons_ets, facecolor='none', edgecolors=color_ets, linewidths=thickness)
    ax.add_collection(p_ets)

    p_golf = PatchCollection(
        polygons_golf, facecolor='none', edgecolors=color_golf, linewidths=thickness)
    ax.add_collection(p_golf)

    p_gtf = PatchCollection(
        polygons_gtf, facecolor='none', edgecolors=color_gtf, linewidths=thickness)
    ax.add_collection(p_gtf)

    p_har = PatchCollection(
        polygons_har, facecolor='none', edgecolors=color_har, linewidths=thickness)
    ax.add_collection(p_har)

    p_over = PatchCollection(
        polygons_over, facecolor='none', edgecolors=color_over, linewidths=thickness)
    ax.add_collection(p_over)

    p_sta = PatchCollection(
        polygons_sta, facecolor='none', edgecolors=color_sta, linewidths=thickness)
    ax.add_collection(p_sta)

    p_tenn = PatchCollection(
        polygons_tenn, facecolor='none', edgecolors=color_tenn, linewidths=thickness)
    ax.add_collection(p_tenn)

    p_train = PatchCollection(
        polygons_train, facecolor='none', edgecolors=color_train, linewidths=thickness)
    ax.add_collection(p_train)

    p_sh = PatchCollection(
        polygons_sh, facecolor='none', edgecolors=color_sh, linewidths=thickness)
    ax.add_collection(p_sh)

    p_pe = PatchCollection(
        polygons_pe, facecolor='none', edgecolors=color_pe, linewidths=thickness)
    ax.add_collection(p_pe)

    p_br = PatchCollection(
        polygons_br, facecolor='none', edgecolors=color_br, linewidths=thickness)
    ax.add_collection(p_br)

    p_sw = PatchCollection(
        polygons_sw, facecolor='none', edgecolors=color_sw, linewidths=thickness)
    ax.add_collection(p_sw)

    p_wm = PatchCollection(
        polygons_wm, facecolor='none', edgecolors=color_wm, linewidths=thickness)
    ax.add_collection(p_wm)

    p_st = PatchCollection(
        polygons_st, facecolor='none', edgecolors=color_st, linewidths=thickness)
    ax.add_collection(p_st)

    p_veh = PatchCollection(
        polygons_veh, facecolor='none', edgecolors=color_veh, linewidths=thickness)
    ax.add_collection(p_veh)


    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0,
                         gt_bbox_color=(255, 102, 61),
                         gt_text_color=(255, 102, 61),
                         gt_mask_color=(255, 102, 61),
                         det_bbox_color=(72, 101, 241),
                         det_text_color=(72, 101, 241),
                         det_mask_color=(72, 101, 241),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=True,
                         wait_time=0,
                         out_file=None):
    """General visualization GT and result function.

    Args:
      img (str or ndarray): The image to be displayed.)
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'
      result (tuple[list] or list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown.  Default: 0
      gt_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (255, 102, 61)
      det_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (72, 101, 241)
      thickness (int): Thickness of lines. Default: 2
      font_size (int): Font size of texts. Default: 13
      win_name (str): The window name. Default: ''
      show (bool): Whether to show the image. Default: True
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
         Default: None

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(
        result,
        (tuple, list)), f'Expected tuple or list, but get {type(result)}'

    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    img = mmcv.imread(img)

    img = imshow_det_bboxes(
        img,
        annotation['gt_bboxes'],
        annotation['gt_labels'],
        gt_masks,
        class_names=class_names,
        bbox_color=gt_bbox_color,
        text_color=gt_text_color,
        mask_color=gt_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False)

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        segms = mask_util.decode(segms)
        segms = segms.transpose(2, 0, 1)

    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms=segms,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color=det_bbox_color,
        text_color=det_text_color,
        mask_color=det_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    return img
