import numpy as np
import cv2


# Get surface plane from the IMU
def plane_from_IMU(roll, pitch, height):
    # roll and pitch are in degrees, while height is in meters

    pitch = -pitch
    roll = -roll

    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    n = np.array([[0], [0], [1]])
    n = np.dot(np.dot(Rx, Ry), n)

    B = np.append(n, height)
    B = B / np.linalg.norm(B)

    return B


# Ganare danger zone binary mask based on estimated ground plane, height of camera and specified danger-range
def danger_zone_to_mask(roll, pitch, height, rnge, M, D, w, h):
    # roll in pitch (in degrees)
    # height (height of a camera above the water surface - in our case cca 0.7m)
    # M and D are calibration matrix and vector of distortion coefficients, respectively
    mask = np.zeros([h, w], dtype=np.uint8)

    A, B, C, D = plane_from_IMU(roll, pitch, height)

    N = 1000
    r = np.linspace(0, 180, N)
    x = np.sin(np.radians(r)) * rnge
    y = np.cos(np.radians(r)) * rnge
    z = np.zeros(N) + (-1 / C)
    z = z * (A * x + B * y + D)

    points = np.transpose(np.array([-y, -z, x]))
    pp, _ = cv2.projectPoints(points, np.identity(3), np.zeros([1, 3]), M, distCoeffs=D)

    poly = []

    for p in pp:
        x = p[0, 0]
        y = p[0, 1]
        if 0 < x <= w and 0 < y <= h:
            poly.append([int(x), int(y)])

    poly.insert(0, [0, poly[0][1]])
    poly.insert(0, [0, h])
    poly.append([w, poly[-1][1]])
    poly.append([w, h])
    poly.append([0, h])

    cv2.fillPoly(mask, np.array([poly]), color=255)

    return mask
