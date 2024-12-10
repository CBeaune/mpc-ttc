import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians

# Fonction pour calculer les sommets d'un rectangle orienté
def get_corners(center, width, height, angle):
    # angle = radians(angle)
    cx, cy = center
    dx = height / 2
    dy = width / 2

    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])

    corners = [
        np.array([dx, dy]),
        np.array([-dx, dy]),
        np.array([-dx, -dy]),
        np.array([dx, -dy])
    ]
    corners = [rotation_matrix @ corner + np.array([cx, cy]) for corner in corners]
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
                closest_pair = (p2, proj)
                projection = proj
    
    return closest_pair, min_dist, projection