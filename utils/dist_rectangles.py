import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians

# Fonction pour calculer les sommets d'un rectangle orienté
def get_corners(center, width, height, angle):
    angle = radians(angle)
    cx, cy = center
    dx = width / 2
    dy = height / 2

    corners = [
        (cx + cos(angle) * dx - sin(angle) * dy, cy + sin(angle) * dx + cos(angle) * dy),
        (cx - cos(angle) * dx - sin(angle) * dy, cy - sin(angle) * dx + cos(angle) * dy),
        (cx - cos(angle) * dx + sin(angle) * dy, cy - sin(angle) * dx - cos(angle) * dy),
        (cx + cos(angle) * dx + sin(angle) * dy, cy + sin(angle) * dx - cos(angle) * dy)
    ]
    return np.array(corners)

# Fonction pour calculer la distance euclidienne entre deux points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Fonction pour calculer la distance d'un point à une arête (segment AB)
def point_to_segment(p, a, b):
    # Vecteurs
    ab = np.array(b) - np.array(a)
    ap = np.array(p) - np.array(a)
    
    # Projection scalaire de ap sur ab
    t = np.dot(ap, ab) / np.dot(ab, ab)
    
    # Si la projection est en dehors du segment, retourner la distance au point le plus proche (A ou B)
    if t < 0.0:
        return distance(p, a), a
    elif t > 1.0:
        return distance(p, b), b
    
    # Projection sur le segment (calcul de la position exacte sur le segment)
    projection = np.array(a) + t * ab
    return distance(p, projection), projection

# Fonction pour trouver la paire de points la plus proche entre deux rectangles, y compris sommets et arêtes
def closest_points(rect1, rect2):
    min_dist = float('inf')
    closest_pair = None
    projection = None
    
    # Comparer chaque sommet du rectangle 1 avec chaque arête du rectangle 2
    for p1 in rect1:
        for i in range(4):
            p2, p3 = rect2[i], rect2[(i + 1) % 4]
            dist, proj = point_to_segment(p1, p2, p3)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (p1, proj)
                projection = proj
    
    # Comparer chaque sommet du rectangle 2 avec chaque arête du rectangle 1
    for p2 in rect2:
        for i in range(4):
            p1, p3 = rect1[i], rect1[(i + 1) % 4]
            dist, proj = point_to_segment(p2, p1, p3)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (proj, p2)
                projection = proj
    
    return closest_pair, min_dist, projection

def calc_distance(data, ego_robot, other_robot, dim):
    center1 = (data[ego_robot]['x'], data[ego_robot]['y'])
    center2 = (data[other_robot]['x'], data[other_robot]['y'])
    width1, height1, angle1 = 0.178, 0.178, data[ego_robot]['theta']
    width2, height2, angle2 = 0.178, 0.178, data[other_robot]['theta']

    rect1 = get_corners(center1, width1, height1, angle1)
    rect2 = get_corners(center2, width2, height2, angle2)
    closest_pair, min_dist, projection = closest_points(rect1, rect2)
    return min_dist

def min_distance_metrics(data, other, t, geometries: dict):

    assert other in data.columns, f"{other} not in data"
    assert t < len(data['tb0_0']['x_noisy']), f"t={t} is out of bounds"
    assert other in geometries, f"{other} not in geometries"
    assert geometries[other]['type'] == 'rectangle', f"{other} is not a rectangle"

    center1 = (data['tb0_0']['x_noisy'][t], data['tb0_0']['y_noisy'][t])
    angle1 = data['tb0_0']['theta_noisy'][t]
    width1, height1 = geometries['tb0_0']['width'], geometries['tb0_0']['height']

    center2 = (data[other]['x_noisy'][t], data[other]['y_noisy'][t])
    angle2 = data[other]['theta_noisy'][t]
    width2, height2 = geometries[other]['width'], geometries[other]['height']

    rect1 = get_corners(center1, width1, height1, angle1)
    rect2 = get_corners(center2, width2, height2, angle2)
    _, min_dist, _ = closest_points(rect1, rect2)
    return min_dist

def gt_distance(data, other, t, geometries: dict):

    assert other in data.columns, f"{other} not in data"
    assert t < len(data['tb0_0']['x']), f"t={t} is out of bounds"
    assert other in geometries, f"{other} not in geometries"
    assert geometries[other]['type'] == 'rectangle', f"{other} is not a rectangle"

    center1 = (data['tb0_0']['x'][t], data['tb0_0']['y'][t])
    angle1 = data['tb0_0']['theta'][t]
    width1, height1 = geometries['tb0_0']['width'], geometries['tb0_0']['height']

    center2 = (data[other]['x'][t], data[other]['y'][t])
    angle2 = data[other]['theta'][t]
    width2, height2 = geometries[other]['width'], geometries[other]['height']

    rect1 = get_corners(center1, width1, height1, angle1)
    rect2 = get_corners(center2, width2, height2, angle2)
    _, min_dist, _ = closest_points(rect1, rect2)
    return min_dist
    