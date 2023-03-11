import cv2
import math
import numpy as np
from PIL import Image
from noise import pnoise3
import depth_map as dm

def get_image_info(src):
    im = Image.open(src)
    try:
        dpi =  im.info['dpi']
    except:
        dpi = (80,80)
    
    return dpi

def get_alti_dist_angle(img, path, depth, vertical_fov, horizontal_angle, camera_altitude):
    img_dpi = get_image_info(path)
    height, width = img.shape[:2]
    altitude = np.empty((height, width))
    distance = np.empty((height, width))
    angle = np.empty((height, width))
    depth_min = depth.min()

    for j in range(width):
        for i in range(height):
            theta = i / (height - 1) * vertical_fov

            horizontal_angle = 0

            if horizontal_angle == 0:
                if theta < 0.5 * vertical_fov:
                    distance[i, j] = depth[i, j] / math.cos(math.radians(0.5 * vertical_fov - theta))
                    h_half = math.tan(0.5*vertical_fov)*depth_min
                    y2 = (0.5*height-i)/img_dpi[0]*2.56
                    y1 = h_half*y2/(height/img_dpi[0]*2.56)

                    altitude[i, j] = camera_altitude+depth[i, j]*y1/depth_min
                    angle[i, j] = 0.5 * vertical_fov - theta
                elif theta == 0.5 * vertical_fov:
                    distance[i, j] = depth[i, j]
                    h_half = math.tan(0.5*vertical_fov)*depth_min
                    y2 = (i-0.5*height)/img_dpi[0]*2.56
                    y1 = h_half*y2/(height/img_dpi[0]*2.56)

                    altitude[i, j] = max(camera_altitude - depth[i, j]*y1/depth_min, 0)
                    angle[i, j] = 0
                elif theta > 0.5 * vertical_fov:
                    distance[i, j] = depth[i, j] / math.cos(math.radians(theta-0.5*vertical_fov))
                    h_half = math.tan(0.5*vertical_fov)*depth_min
                    y2 = (i-0.5*height)/img_dpi[0]*2.56
                    y1 = h_half*y2/(height/img_dpi[0]*2.56)

                    altitude[i, j] = max(camera_altitude - depth[i, j]*y1/depth_min, 0)
                    angle[i, j] = -(theta - 0.5 * vertical_fov)

    return altitude, distance, angle

def get_perlin_noise(img, depth):
    p1 = Image.new('L', (img.shape[1], img.shape[0]))
    p2 = Image.new('L', (img.shape[1], img.shape[0]))
    p3 = Image.new('L', (img.shape[1], img.shape[0]))

    scale = 1/130.0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v+1)*128.0)
            p1.putpixel((x, y), color)

    scale = 1/60.0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v+0.5)*128)
            p2.putpixel((x, y), color)

    scale = 1/10.0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v+1.2)*128)
            p3.putpixel((x, y), color)

    perlin = (np.array(p1) + np.array(p2)/2 + np.array(p3)/4)/3

    return perlin

def imHaze(path, intensity=3, CAMERA_ALTITUDE=3.5, HORIZONTAL_ANGLE=0, CAMERA_VERTICAL_FOV=64):
    # load rgb image
    img = cv2.imread(path)
    h,w,_ = img.shape
    img = cv2.resize(img, dsize=(int(w/2), int(h/2)),interpolation=cv2.INTER_AREA)
    # gen depth map
    depthmap = dm.gen_depthmap(img)
    dmin = depthmap.min()
    cv2.imwrite('dimg.jpg', depthmap)
    depth = (depthmap.astype(np.float64)) * intensity
    depth = np.array([[255 if x >255 else x for x in xx] for xx in depth])

    I = np.empty_like(img)

    # init constatns
    # atmosphere
    VISIBILITY_RANGE_MOLECULE = 12  # m
    VISIBILITY_RANGE_AEROSOL = 450  # m
    ECM = 3.912 / VISIBILITY_RANGE_MOLECULE  # EXTINCTION_COEFFICIENT_MOLECULE / m
    ECA = 3.912 / VISIBILITY_RANGE_AEROSOL  # EXTINCTION_COEFFICIENT_AEROSOL / m
    FT = 70  # m FOG TOP
    HT = 34  # m HAZE TOP

    elevation, distance, angle = get_alti_dist_angle(img, path, depth,
                                                            CAMERA_VERTICAL_FOV,
                                                            HORIZONTAL_ANGLE,
                                                            CAMERA_ALTITUDE)

    if FT != 0:
        perlin = get_perlin_noise(img, depth)
        # ECA = ECA * np.exp(-elevation/(FT+0.00001))
        c = (1-elevation/(FT+0.00001))
        c[c<0] = 0

        if FT > HT:
            ECM = (ECM * c + (1-c)*ECA) * (perlin/255)
        else:
            ECM = (ECA * c + (1-c)*ECM) * (perlin/255)


    distance_through_fog = np.zeros_like(distance)
    distance_through_haze = np.zeros_like(distance)
    distance_through_haze_free = np.zeros_like(distance)

    if (FT < HT) and (FT != 0):
        idx1 = (np.logical_and(HT > elevation, elevation > FT))
        idx2 = elevation <= FT
        idx3 = elevation >= HT
        if CAMERA_ALTITUDE <= FT:
            distance_through_fog[idx2] = distance[idx2]
            distance_through_haze[idx1] = (elevation[idx1] - FT) * distance[idx1] \
                                            / (elevation[idx1] - CAMERA_ALTITUDE)

            distance_through_fog[idx1] = distance[idx1] - distance_through_haze[idx1]
            distance_through_fog[idx3] = (FT - CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)
            distance_through_haze[idx3] = (HT - FT) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)
            distance_through_haze_free[idx3] = distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]

        elif CAMERA_ALTITUDE > HT:
            distance_through_haze_free[idx3] = distance[idx3]
            distance_through_haze[idx1] = (FT - elevation[idx1]) * distance_through_haze_free[idx1] \
                                        / (CAMERA_ALTITUDE - HT)
            distance_through_haze_free[idx1] = distance[idx1] - distance_through_haze[idx1]


            distance_through_fog[idx2] = (FT - elevation[idx2]) * distance[idx2] / (CAMERA_ALTITUDE - elevation[idx2])
            distance_through_haze[idx2] = (HT - FT) * distance / (CAMERA_ALTITUDE - elevation[idx2])
            distance_through_haze_free[idx2] = distance[idx2] - distance_through_haze[idx2] - distance_through_fog[idx2]

        elif FT < CAMERA_ALTITUDE <= HT:
            distance_through_haze[idx1] = distance[idx1]
            distance_through_fog[idx2] = (FT - elevation[idx2]) * distance[idx2] / (CAMERA_ALTITUDE - elevation[idx2])
            distance_through_haze[idx2] = distance[idx2] - distance_through_fog[idx2]
            distance_through_haze_free[idx3] = (elevation[idx3] - HT) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)
            distance_through_haze[idx3] = (HT - CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)

        I[:, :, 0] = img[:, :, 0] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
        I[:, :, 1] = img[:, :, 1] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
        I[:, :, 2] = img[:, :, 2] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
        O = 1-np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)

    elif FT > HT:
        if CAMERA_ALTITUDE <= HT:
            idx1 = (np.logical_and(FT > elevation, elevation > HT))
            idx2 = elevation <= HT
            idx3 = elevation >= FT

            distance_through_haze[idx2] = distance[idx2]
            distance_through_fog[idx1] = (elevation[idx1] - HT) * distance[idx1] \
                                            / (elevation[idx1] - CAMERA_ALTITUDE)
            distance_through_haze[idx1] = distance[idx1] - distance_through_fog[idx1]
            distance_through_haze[idx3] = (HT - CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)
            distance_through_fog[idx3] = (FT - HT) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)
            distance_through_haze_free[idx3] = distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]

        elif CAMERA_ALTITUDE > FT:
            idx1 = (np.logical_and(HT > elevation, elevation > FT))
            idx2 = elevation <= FT
            idx3 = elevation >= HT

            distance_through_haze_free[idx3] = distance[idx3]
            distance_through_haze[idx1] = (FT - elevation[idx1]) * distance_through_haze_free[idx1] \
                                        / (CAMERA_ALTITUDE - HT)
            distance_through_haze_free[idx1] = distance[idx1] - distance_through_haze[idx1]
            distance_through_fog[idx2] = (FT - elevation[idx2]) * distance[idx2] / (CAMERA_ALTITUDE - elevation[idx2])
            distance_through_haze[idx2] = (HT - FT) * distance / (CAMERA_ALTITUDE - elevation[idx2])
            distance_through_haze_free[idx2] = distance[idx2] - distance_through_haze[idx2] - distance_through_fog[idx2]

        elif HT < CAMERA_ALTITUDE <= FT:
            idx1 = (np.logical_and(HT > elevation, elevation > FT))
            idx2 = elevation <= FT
            idx3 = elevation >= HT

            distance_through_haze[idx1] = distance[idx1]
            distance_through_fog[idx2] = (FT - elevation[idx2]) * distance[idx2] / (CAMERA_ALTITUDE - elevation[idx2])
            distance_through_haze[idx2] = distance[idx2] - distance_through_fog[idx2]
            distance_through_haze_free[idx3] = (elevation[idx3] - HT) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)
            distance_through_haze[idx3] = (HT - CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - CAMERA_ALTITUDE)

        I[:, :, 0] = img[:, :, 0] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
        I[:, :, 1] = img[:, :, 1] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
        I[:, :, 2] = img[:, :, 2] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
        O = 1-np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)

    else:
        assert(False)

    fog = np.empty_like(img)  # fog color buffer
    fog[:, :, 0] = 225
    fog[:, :, 1] = 225
    fog[:, :, 2] = 200

    result = np.empty_like(img)
    result[:, :, 0] = I[:, :, 0] + O * fog[:, :, 0]
    result[:, :, 1] = I[:, :, 1] + O * fog[:, :, 1]
    result[:, :, 2] = I[:, :, 2] + O * fog[:, :, 2]

    result = cv2.resize(result, dsize=(w,h), interpolation=cv2.INTER_AREA)
    return result


if __name__ == "__main__":
    result = imHaze("test.jpg")
    cv2.imwrite('result.jpg', result)
    pass