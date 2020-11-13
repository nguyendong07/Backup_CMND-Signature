from pdf2image import convert_from_path
import time
from PIL import Image
ouputDir = ''
pages = convert_from_path('17020669_DinhTienDat.pdf')


for page in pages:
  myfile = ouputDir + 'pdf.jpg'
  page.save(myfile)
