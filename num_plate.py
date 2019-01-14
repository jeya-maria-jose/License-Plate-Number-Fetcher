import cv2
import cv2
import pytesseract
from PIL import Image
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
txt = raw_input("Type the image's folder directory: ")
img = cv2.imread(txt)

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


# iterative bilateral filter(removes noise while preserving edges)
noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
# Histogram equalisation 
equal_histogram = cv2.equalizeHist(noise_removal)

# Morphological opening with a rectangular structure element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)
sub_morp_image = cv2.subtract(equal_histogram,morph_image)
ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
canny_image = cv2.Canny(thresh_image,250,255)
canny_image = cv2.convertScaleAbs(canny_image)

# dilation to strengthen the edges
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
for c in contours:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
	if len(approx) == 4:  # Select the contour with 4 corners
		screenCnt = approx

		break
final = cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
mask = np.zeros(img_gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)
y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb))
y = cv2.equalizeHist(y)
final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCrCb2RGB)
cv2.imwrite('C:/Python27/temp.jpg',final_image)

#result = pytesseract.image_to_string(Image.fromarray(new_image))
ans = image_to_string(Image.open('C:/Python27/temp.jpg'), lang='eng')
file = open("number.txt","w") 
 
file.write(ans)

file.close()
cv2.waitKey() # Wait for a keystroke from the user
