import torch
import numpy as np
import cv2
from detecto import core, utils, visualize
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time, os
# merge overlap boxes
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
    idxs = np.argsort(area)

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


# return final result



# find centerpoint
def get_center_point(box):
    xmin, ymin, xmax, ymax = box
    return (xmin + xmax) // 2, (ymin + ymax) // 2


#final_points = list(map(get_center_point, final_boxes))
#label_boxes = dict(zip(final_labels, final_points))


def rotate(coor):
  br,bl,tr,tl = [coor['bottom_right'],coor['bottom_left'],coor['top_right'],coor['top_left']]
  #print(br[0])
  dis1=euclidean(coor['bottom_right'],coor['emblem']) # kc euclid tu bottom right den emblem
  dis2=euclidean(coor['bottom_left'],coor['emblem']) # kc euclid tu bottom left den emblem
  dis3=euclidean(coor['top_right'],coor['emblem'])
  dis4=euclidean(coor['top_left'],coor['emblem'])
  a=[dis1,dis2,dis3,dis4]
  index= 0
  minCoor = a[0]
  i = 0
  # tim khoang cach ngan nhat tu quoc huy(emblem) => cac toa do goc/ output = index
  while(i<len(a)):
    if(minCoor > a[i]):
      minCoor = a[i]
      index = i
    i= i+1
  # index = 0,1,2,3: quoc huy o goc: br,bl,tr,tl
  #print(index)
  if (index==0 ):
    return np.float32([br,bl,tl,tr])
  elif (index==1):
    return np.float32([bl,tl,tr,br])
  elif (index==2):
    return np.float32([tr,br,bl,tl])
  else: return np.float32([tl,tr,br,bl])

#crop image
def perspective_transoform(image, source_points):
    #print(source_points)
    dest_points = np.float32([[0,0], [500,0], [500,300], [0,300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))
    return dst
# return image after crop
#source_points = rotate(label_boxes)
# Transform
#crop = perspective_transoform(image, source_points)




real_h= 24.5
real_w = 15.5
# crop text area
def CropTextArea(image,coordinate,H,W):
  return image[coordinate[0]:coordinate[0]+W,coordinate[1]:coordinate[1]+H]

# caculate frames of text  
def CaculateNameFrame(h,w):
  coordinate = (np.int32((5.5/real_w)*w),np.int32((10.5/real_h)*h))
  width = np.int32(2.5/real_w*w )
  height= np.int32(11/real_h *h )
  return (coordinate, height, width)
def CaculateIdFrame(h,w):
  coordinate = (np.int32((4/real_w)*w),np.int32((10.5/real_h)*h))
  width = np.int32(2/real_w*w )
  height= np.int32( 11/real_h *h )
  return (coordinate, height, width)
def CaculateDateFrame(h,w):
  coordinate = (np.int32((8/real_w)*w),np.int32((12/real_h)*h))
  width = np.int32(1.5/real_w*w )
  height= np.int32( 11/real_h *h )
  return (coordinate, height, width)
def CaculateAddress1Frame(h,w):
  coordinate = (np.int32((10.5/real_w)*w),np.int32((11.5/real_h)*h))
  width = np.int32(2.5/real_w*w )
  height= np.int32( 12/real_h *h )
  return (coordinate, height, width)
def CaculateAddress2Frame(h,w):
  coordinate = (np.int32((12.5/real_w)*w),np.int32((11.5/real_h)*h))
  width = np.int32(2.5/real_w*w )
  height= np.int32( 12/real_h *h )
  return (coordinate, height, width)

#(coordinate, height, width) = CaculateIdFrame(iHeight,iWidth)
#(coordinate3, height3, width3) = CaculateNameFrame(iHeight,iWidth)
#(coordinate4, height4, width4) = CaculateDateFrame(iHeight,iWidth)
#(coordinate5, height5, width5) = CaculateAddress1Frame(iHeight,iWidth)
#(coordinate6, height6, width6) = CaculateAddress2Frame(iHeight,iWidth)

#image1 = CropTextArea(crop,coordinate, height, width)
#image3 = CropTextArea(crop,coordinate3, height3, width3)
#image4 = CropTextArea(crop,coordinate4, height4, width4)
#image5 = CropTextArea(crop,coordinate5, height5, width5)
#image6 = CropTextArea(crop,coordinate6, height6, width6)

config = Cfg.load_config_from_name('vgg_seq2seq')
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
detector = Predictor(config)
def box2text(path):
  img = Image.open(path)
  res = detector.predict(img)
  os.remove(path)
  return res
  
def check_enough_labels(labels):
  tmp = ['emblem','top_left','top_right','bottom_left','bottom_right']
  for i in tmp:
    bool = i in labels
    if bool == False: 
      return(False)
  return(True)
  
def checktypeimage(path):
  imgName = str(path)
  imgName = imgName[imgName.rindex('.')+1:]
  imgName = imgName.lower()
  return imgName
  
def Main(path):
  model = torch.load("/home/tandanml/ML/ExtractInfor/ExText/DetectCMND311")
  typeimgae = checktypeimage(path) 
  if(typeimgae!='png' and typeimgae!='jpeg' and typeimgae!='jpg'):
    return ({"code" : "408", "message": "Anh khong dung dinh dang"})
  else:
    image = np.asarray(bytearray(path.read()), dtype='uint8')
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)
    imgName = str(path)
    imgName= imgName[:imgName.rindex('.')]
    imgName = 'image/'+imgName +'_'+str(time.time())+'.jpg'
    cv2.imwrite(imgName,image)
    image = utils.read_image(imgName)
    labels, boxes, scores = model.predict(image)
    print(labels)
    if(check_enough_labels(labels)==False):
      return({
        'success':False,
        'code': 400,
        'message':'Khong xac dinh duoc the can cuoc hoac cmnd'
      })
    final_boxes, final_labels = non_max_suppression_fast(boxes.numpy(), labels, 0.15)
    if(check_enough_labels(final_labels)==False):
      return({
        'success':False,
        'code': 404,
        'message':'Chat luong anh thap hoac khong du thong tin'
      })
    final_points = list(map(get_center_point, final_boxes))
    label_boxes = dict(zip(final_labels, final_points))
    
    source_points = rotate(label_boxes)
    crop = perspective_transoform(image, source_points)
    iHeight = crop.shape[1]
    iWidth = crop.shape[0]
    (coordinate, height, width) = CaculateIdFrame(iHeight,iWidth)
    (coordinate3, height3, width3) = CaculateNameFrame(iHeight,iWidth)
    (coordinate4, height4, width4) = CaculateDateFrame(iHeight,iWidth)
    (coordinate5, height5, width5) = CaculateAddress1Frame(iHeight,iWidth)
    (coordinate6, height6, width6) = CaculateAddress2Frame(iHeight,iWidth)
    # extract id field
    image1 = CropTextArea(crop,coordinate, height, width)
    image= cv2.imwrite('name.jpg',image1)
    id_field = box2text('name.jpg')
    # name field
    image3 = CropTextArea(crop,coordinate3, height3, width3)
    image= cv2.imwrite('name.jpg',image3)
    name = box2text('name.jpg')
    # date field
    image4 = CropTextArea(crop,coordinate4, height4, width4)
    date= cv2.imwrite('name.jpg',image4)
    date = box2text('name.jpg')
    # address 1 field
    image5 = CropTextArea(crop,coordinate5, height5, width5)
    address1= cv2.imwrite('name.jpg',image5)
    address1= box2text('name.jpg')
    # address 2 field
    #image6 = CropTextArea(crop,coordinate6, height6, width6)
    #address2= cv2.imwrite('name.jpg',image6)
    #address2 = box2text('name.jpg')
    
    return({
      'success': True,
      'code': 200,
      'message':'Thanh cong',
      'data':{
        "id":id_field,
        "name": name,
        "date":date,
        "address":address1
      }
    })
    