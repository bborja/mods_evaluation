import numpy as np
import cv2


# Get Rimu matrix (see RAS paper)
def get_Rimu(roll, pitch):
    # roll and pitch are in degrees
    # Roll (rotation around Z axis)
    # Pitch (rotation around X axis)
    # Z points towards sailing direction and Y is vertical

    # Roll
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

    # Yaw
    Ry = np.eye(3)

    # Pitch
    c, s = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    # Rimu
    R = np.matmul(np.linalg.inv(Rz), np.matmul(Ry, Rx))

    return R


# Generate danger zone binary mask based on estimated ground plane, height of camera and specified danger-range
def danger_zone_to_mask(roll, pitch, height, rnge, M, DC, w, h):
    # roll and pitch (in degrees)
    # height (height of a camera above the water surface - in our case cca 0.7m - 1.0m)
    # M and DC are calibration matrix and vector of distortion coefficients, respectively
    # w, h are width and height of the image

    mask = np.zeros([h, w], dtype=np.uint8)

    # Margin angle within which we generate points
    margin_angle = 40

    R_imu = get_Rimu(roll, pitch)

    N = 100
    r = np.linspace(0 + margin_angle, 180 - margin_angle, N)
    z = np.sin(np.radians(r)) * rnge
    x = -np.cos(np.radians(r)) * rnge
    y = np.zeros(N)

    points = np.transpose(np.matmul(R_imu, np.array([x, y, z])))

    pp, _ = cv2.projectPoints(points, np.eye(3), np.array([[0, height, 0]]), M, distCoeffs=DC)

    poly = []

    for p in pp:
        x = int(p[0, 0])
        y = int(p[0, 1])
        if 0 < x <= w and 0 < y <= h:
            poly.append([x, y])

    poly.insert(0, [0, poly[0][1]])
    poly.insert(0, [0, h])
    poly.append([w, poly[-1][1]])
    poly.append([w, h])
    poly.append([0, h])

    cv2.fillPoly(mask, np.array([poly]), color=255)

    return mask
