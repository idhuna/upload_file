import numpy as np
import cv2
import os
# from matplotlib import pyplot as plt


def avgColor(img, thredL=0, thredU=256):
    np.seterr(divide='ignore', invalid='ignore')
    avg_img = np.array(img)
    avg_img = avg_img.astype('float')
    r = avg_img[:,:,0]
    g = avg_img[:,:,1]
    b = avg_img[:,:,2]
    avg_color = [np.average(r[r!=0]),
                 np.average(g[g!=0]),
                 np.average(b[b!=0])]
#     avg_img[(avg_img == 0) | (avg_img < thredL) | (thredU < avg_img)] = np.nan
#     avg_img_row = np.nanmean(avg_img, axis=0)
#     avg_color = np.nanmean(avg_img_row, axis=0)
    return np.array(avg_color)
 
def cropBankNote(img):
    bank_note = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 
    bank_note = cv2.medianBlur(bank_note,13)
    threshold = cv2.threshold(bank_note,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations = 1)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations = 1)

    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
 
    font = cv2.FONT_HERSHEY_COMPLEX
    bank_size = 1000000
    isFoundBank = False
    bank_notes = []
    approxs = []
    rank = float('inf')
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.002*cv2.arcLength(cnt, True), True)
        
        if  4<= len(approx) <= 10 and  (2/3)*bank_size <= cv2.contourArea(cnt) <= (4/3)*bank_size:
            
            new_rank = abs(8 - len(approx)) + abs(1000000-cv2.contourArea(cnt))//100000
            if rank > new_rank:
                rank = new_rank
            approxs = [approx,len(approx),cv2.contourArea(cnt)]
            isFoundBank = True

    if isFoundBank:   
        print('Found Bank Note')
        # Apply mask on imge
        approx_selected, approx_len, approx_area = approxs
        print('Shape :', approx_len, ', Area :', approx_area)
        mask = np.zeros(bank_note.shape, np.uint8)
        cv2.drawContours(mask,[approx_selected],0,255,-1)
        result = img.copy()
        result[mask==0] = (0)
 
        return avgColor(np.array(result))
    else:
        print('Not Found Bank Note')
        return np.array([False])

def colorCorrection(img):
    avgBankNote = cropBankNote(img)
    if avgBankNote.all():
        RGBRaw = np.array(avgBankNote)
    else:
        return img
        
    RGBBase = np.array([1.0, 0.97689466, 0.96918224])
    RGBRaw = RGBRaw/RGBRaw[0]
    RGBFactor = RGBBase/RGBRaw
    imgCC = np.array(img)
    # RGB
    red = np.copy(imgCC[:,:,0])
    greenFactor = RGBFactor[1]
    green = imgCC[:,:,1] * greenFactor
    blueFactor = RGBFactor[2]
    blue = imgCC[:,:,2] * blueFactor
    # Green
    imgCC[:,:,1] = np.where(blue > 255, (255/blue)*green, green)
    imgCC[:,:,1] = np.where((green > 255) & (green >= blue),255, imgCC[:,:,1])
    # Blue
    imgCC[:,:,2] = np.where(green > 255, (255/green)*blue, blue)
    imgCC[:,:,2] = np.where((blue > 255) & (blue >= green),255, imgCC[:,:,2])
    # Red
    imgCC[:,:,0] = np.where((green > 255) | (blue > 255), 
                            np.where(green > blue, (255/green)*red , (255/blue)*red), red)
    avgAfterCC = avgColor(imgCC)
    avgBeforeCC = avgColor(img)
    print('RGBratioBeforeCC =',avgBeforeCC/avgBeforeCC[0])
    print('RGBratioAfterCC =',avgAfterCC/avgAfterCC[0])
    
    return imgCC

def rotate_image(mat, angle):

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def pred_rice(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    write_path = path[:-4]+"test"+path[-4:]
    print('write_path',write_path)
    cv2.imwrite(write_path, img)
    return write_path[-22:]

def rice_detection(path, filename, crops):
    print('Read image at', path)
    img = cv2.imread(path)
    y, x, w, h, rotate = crops
    print(x,y,w,h)
    img = rotate_image(img, rotate)
    img = img[x:h,y:w]
    print(img.shape,rotate)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colorCorrection(img)
    return pred_rice(path, img)

def main():
    pass

if __name__ == "__main__":
    main()