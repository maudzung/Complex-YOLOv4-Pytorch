import torch
from math import pi


def _angle_to_point(point, centre):
    '''calculate angle in 2-D between points and x axis'''
    delta = point - centre
    res = torch.atan(delta[1] / delta[0])
    if delta[0] < 0:
        res += pi
    return res


def area_of_triangle(p1, p2, p3):
    '''calculate area of any triangle given co-ordinates of the corners'''
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return torch.norm(cross) / 2.


def convex_hull(points):
    """Calculate subset of points that make a convex hull around points
    Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

    :Parameters:
        points : (m, 2) array of points for which to find hull
    :Returns:
        hull_points : ndarray (n x 2) convex hull surrounding points
    """

    n_pts = points.size(0)
    assert (n_pts >= 4)
    centre = points.mean(0)
    # angles = torch.apply_along_axis(_angle_to_point, 0, points, centre)
    angles = torch.stack([_angle_to_point(point, centre) for point in points], dim=0)
    pts_ord = points[angles.argsort(), :]
    pts = [x[0] for x in zip(pts_ord)]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i], pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i], pts[(i + 2) % n_pts])
            if Aij + Ajk < Aik:
                del pts[i + 1]
            i += 1
            n_pts = len(pts)
        k += 1
    return torch.stack(pts)


if __name__ == "__main__":
    points = torch.rand((40, 2)).cuda()
    hull_pts = convex_hull(points)
    print('hull_pts: {}'.format(hull_pts))
