#!/usr/bin/python

import pygame
import json
import math

"""
how many pixel = actual distance in cm
30px = 50cm --> 50/30 = MAP_SIZE_COEFF
"""
MAP_SIZE_COEFF = 1.66666667
GRID_SIZE = 30  # Grid square size in pixels (modifiable)

pygame.init()
screen = pygame.display.set_mode([609, 609])
pygame.display.set_caption("Waypoint Grid")
screen.fill((255, 255, 255))
running = True


class Background(pygame.sprite.Sprite):
    def __init__(self, image, location, scale):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(image)
        self.image = pygame.transform.rotozoom(self.image, 0, scale)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location


def draw_grid(surface, grid_size):
    """
    Draw a grid on the surface.
    """
    for x in range(0, surface.get_width(), grid_size):
        pygame.draw.line(surface, (200, 200, 200), (x, 0), (x, surface.get_height()), 1)
    for y in range(0, surface.get_height(), grid_size):
        pygame.draw.line(surface, (200, 200, 200), (0, y), (surface.get_width(), y), 1)


def snap_to_grid(pos, grid_size):
    """
    Snap a position to the nearest grid intersection.
    """
    x, y = pos
    snapped_x = round(x / grid_size) * grid_size
    snapped_y = round(y / grid_size) * grid_size
    return snapped_x, snapped_y


def get_dist_btw_pos(pos0, pos1):
    """
    Get distance between 2 mouse positions.
    """
    x = abs(pos0[0] - pos1[0])
    y = abs(pos0[1] - pos1[1])
    dist_px = math.hypot(x, y)
    dist_cm = dist_px * MAP_SIZE_COEFF
    return int(dist_cm), int(dist_px)


def get_angle_btw_line(pos0, pos1, posref):
    """
    Get angle between two lines respective to 'posref'
    NOTE: using dot product calculation.
    """
    ax = posref[0] - pos0[0]
    ay = posref[1] - pos0[1]
    bx = posref[0] - pos1[0]
    by = posref[1] - pos1[1]
    # Get dot product of pos0 and pos1.
    _dot = (ax * bx) + (ay * by)
    # Get magnitude of pos0 and pos1.
    _magA = math.sqrt(ax**2 + ay**2)
    _magB = math.sqrt(bx**2 + by**2)
    _rad = math.acos(_dot / (_magA * _magB))
    # Angle in degrees.
    angle = (_rad * 180) / math.pi
    return int(angle)


"""
Main capturing mouse program.
"""
# Load background image.
try:
    bground = Background('image.png', [0, 0], 1.6)
    screen.blit(bground.image, bground.rect)
except FileNotFoundError:
    print("Background image not found. Ensure 'image.png' exists.")
    bground = None

path_wp = []
index = 0
while running:
    # Draw grid and background
    screen.fill((255, 255, 255))  # Clear screen
    if bground:
        screen.blit(bground.image, bground.rect)
    draw_grid(screen, GRID_SIZE)

    # Draw existing waypoints and lines
    for i in range(1, len(path_wp)):
        pygame.draw.line(screen, (255, 0, 0), path_wp[i - 1], path_wp[i], 2)
        pygame.draw.circle(screen, (0, 0, 255), path_wp[i], 5)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Snap position to grid and add to waypoints
            pos = snap_to_grid(pygame.mouse.get_pos(), GRID_SIZE)
            path_wp.append(pos)
            index += 1

    pygame.display.update()

"""
Compute the waypoints (distance and angle).
"""
if path_wp:
    # Append first pos ref. (dummy)
    path_wp.insert(0, (path_wp[0][0], path_wp[0][1] - 10))

    path_dist_cm = []
    path_dist_px = []
    path_angle = []
    for index in range(len(path_wp)):
        # Skip the first and second index.
        if index > 1:
            dist_cm, dist_px = get_dist_btw_pos(path_wp[index - 1], path_wp[index])
            path_dist_cm.append(dist_cm)
            path_dist_px.append(dist_px)

        # Skip the first and last index.
        if index > 0 and index < (len(path_wp) - 1):
            angle = get_angle_btw_line(path_wp[index - 1], path_wp[index + 1], path_wp[index])
            path_angle.append(angle)

    # Print out the information.
    print('path_wp: {}'.format(path_wp))
    print('dist_cm: {}'.format(path_dist_cm))
    print('dist_px: {}'.format(path_dist_px))
    print('dist_angle: {}'.format(path_angle))

    """
    Save waypoints into JSON file.
    """
    waypoints = []
    for index in range(len(path_dist_cm)):
        waypoints.append({
            "dist_cm": path_dist_cm[index],
            "dist_px": path_dist_px[index],
            "angle_deg": 180-path_angle[index]
        })

    # Save to JSON file.
    with open('waypoint.json', 'w') as f:
        path_wp.pop(0)  # Remove the dummy first position
        json.dump({
            "wp": waypoints,
            "pos": path_wp
        }, f, indent=4)
    print("Waypoints saved to 'waypoint.json'.")
else:
    print("No waypoints to save.")

pygame.quit()
