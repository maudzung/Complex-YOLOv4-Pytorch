import sys

import numpy as np
import mayavi.mlab as mlab
import cv2

sys.path.append('../')

from data_process import kitti_data_utils
import config.kitti_config as cnf


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    # draw points
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=None, mode='point', colormap='gnuplot', scale_factor=1,
                  figure=fig)
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_lidar(pc, color=None, fig1=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    # if fig1 is None: fig1 = mlab.figure(figure="point cloud", bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))

    mlab.clf(figure=None)
    if color is None: color = pc[:, 2]
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig1)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)

    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig1)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig1)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig1)

    # draw fov (todo: update to real sensor spec.)
    fov = np.array([  # 45 degree
        [20., 20., 0., 0.],
        [20., -20., 0., 0.],
    ], dtype=np.float64)

    mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig1)
    mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig1)

    # draw square region
    TOP_Y_MIN = -20
    TOP_Y_MAX = 20
    TOP_X_MIN = 0
    TOP_X_MAX = 40
    TOP_Z_MIN = -2.0
    TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig1)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig1)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig1)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig1)

    # mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=60.0, figure=fig1)
    return fig1


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1), line_width=2, draw_text=True, text_scale=(1, 1, 1),
                    color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%d' % n, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=0.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def show_image_with_boxes(img, objects, calib, show3d=False):
    ''' Show image with 2D bounding boxes '''

    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == 'DontCare': continue
        # cv2.rectangle(img2, (int(obj.xmin),int(obj.ymin)),
        #    (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
        box3d_pts_2d, box3d_pts_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = kitti_data_utils.draw_projected_box3d(img2, box3d_pts_2d, cnf.colors[obj.cls_id])
    if show3d:
        cv2.imshow("img", img2)
    return img2


def show_lidar_with_boxes(pc_velo, objects, calib,
                          img_fov=False, img_width=None, img_height=None, fig=None):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''

    if not fig:
        fig = mlab.figure(figure="KITTI_POINT_CLOUD", bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1250, 550))

    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height)

    draw_lidar(pc_velo, fig1=fig)

    for obj in objects:

        if obj.type == 'DontCare': continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)

        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = kitti_data_utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0, 1, 1), line_width=2, draw_text=False)

        mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)

    mlab.view(distance=90)


def merge_rgb_to_bev(img_rgb, img_bev, output_width):
    img_rgb_h, img_rgb_w = img_rgb.shape[:2]
    ratio_rgb = output_width / img_rgb_w
    output_rgb_h = int(ratio_rgb * img_rgb_h)
    ret_img_rgb = cv2.resize(img_rgb, (output_width, output_rgb_h))

    img_bev_h, img_bev_w = img_bev.shape[:2]
    ratio_bev = output_width / img_bev_w
    output_bev_h = int(ratio_bev * img_bev_h)

    ret_img_bev = cv2.resize(img_bev, (output_width, output_bev_h))

    out_img = np.zeros((output_rgb_h + output_bev_h, output_width, 3), dtype=np.uint8)
    # Upper: RGB --> BEV
    out_img[:output_rgb_h, ...] = ret_img_rgb
    out_img[output_rgb_h:, ...] = ret_img_bev

    return out_img
