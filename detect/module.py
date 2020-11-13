import torch
import pickle
import cv2
from detecto import utils
import numpy as np
import os
from pdf2image import convert_from_bytes
from PIL import Image
def non_max_suppression_fast(boxes, labels, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    #
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type

    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")

    return final_boxes, final_labels

# final_boxes, final_labels = non_max_suppression_fast(boxes.numpy(), labels, 0.8)
#find centerpoint
# def get_center_point(box):
#     xmin, ymin, xmax, ymax = box
#     return (xmin + xmax) // 2, (ymin + ymax) // 2
# final_points = list(map(get_center_point, final_boxes))
# label_boxes = dict(zip(final_labels, final_points))
# print(final_labels)
# print(final_boxes)
#
# for i, box in enumerate(final_boxes):
#     box = list(map(int, box))
#     x_min, y_min, x_max, y_max = box
#     x_c = int((x_min+x_max)/2)
#     y_c = int((y_max+y_min)/2)
#     print(x_min, y_min, x_max, y_max)
#     print(x_c, y_c)
#     cv2.circle(image, (x_c,y_c), 20, (255, 0, 0), -1)
#     cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,255,0),2)
#     cv2.putText(image, labels[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
# cv2.imshow('img',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(image.shape)

def main(path):
  byte = bytes(path.read())
  outputDir = 'image_area/'
  pages = convert_from_bytes(byte)
  count = 1
  list1 = []
  for page in pages: 
    filename = outputDir+str(path)[:str(path).rindex('.')] +'_'+str(count)+'.jpg'
    page.save(filename)
    res = detect(filename)
    os.remove(filename)
    if res != None :
      list1.append({"page": count,"data":{
        "toado_x": res[0] + res[2]/2,
        "toado_y": res[1] + res[3]/2
      }})
      print(list1)
    count += 1 
  return list1
  

  
  
def detect(filename):
 model = torch.load("/home/tandanml/ML/ExtractInfor/detect/abc")
 image = utils.read_image(filename)
 labels, boxes, scores = model.predict(image)
 if (len(boxes)==0): 
   return None
 return (boxes[-1])
 #final_boxes, final_labels = non_max_suppression_fast(boxes.numpy(), labels, 0.8)
 #print(final_boxes, final_labels)
 #if final_boxes == []: 
 #return {
 #     "message" : "Khong tim thay vung ky"
 #     }
 #else:
 #   for i, box in enumerate(final_boxes):
 #       box = list(map(int, box))
 #       x_min, y_min, x_max, y_max = box
 #       x_c = int((x_min+x_max)/2)
 #       y_c = int((y_max+y_min)/2)
 #       cv2.circle(image, (x_c, y_c), 20, (255, 0, 0), -1)
  #      cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
  #      cv2.putText(image, labels[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
  #  return {
  #          "Vị trí 4 góc":[x_min, y_min, x_max,y_max],
   #         "ví trí ký:": [x_c, y_c]}

#detect()