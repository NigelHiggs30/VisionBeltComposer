
def boundingBox(img):
    '''
    boundingBox takes N parameters and returns xywh

    :param img: opencv numpy.ndarray of shape (W, H, C)
    :return: x,y,w,h - x and y is the starting point w and h our width and height.
    '''
    
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding on the gray image to create a binary image
    ret,thresh = cv2.threshold(gray,17,255,0)

    # find the contours5
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # take the first contour
    cnt = contours[0]

    # compute the bounding rectangle of the contour
    x,y,w,h = cv2.boundingRect(cnt)

    return x,y,w,h


######################################
# ###########
import cv2
# from PIL import Image

img = cv2.imread(r'C:\Users\nigel\OneDrive\Desktop\OSU\final project\DataCreation\FarwestData\ColabOutput\folder\Original\106.png')

# jug location test data
# jug = img[200:540, 200:500]
x, y, w, h = boundingBox(img)
# Extract the ROI from the image
jug = img[y:y+h, x:x+w]

cv2.imwrite("Img/jug1.jpg",jug)

cv2.imshow("",jug)
cv2.waitKey(0)
cv2.destroyAllWindows()


# im2 = Image.open('C:/Users/nigel/OneDrive/Desktop/OSU/final project/test/Img/jug1.png')
# im2 = im2.resize((150,150))
# im2.save("Img/jug1.jpg",quality=100)


# Load two images
img1 = cv2.imread('C:/Users/nigel/OneDrive/Desktop/OSU/final project/test/Img/EmptyConveyor.jpg')
img2 = cv2.imread('C:/Users/nigel/OneDrive/Desktop/OSU/final project/test/Img/jug1.jpg')

# Define the location where you want to place img2 on img1
start_row = 50  # Replace 100 with the row coordinate
start_col = 90  # Replace 150 with the column coordinate

# I want to put img2 in a specific location, so I create a ROI
rows, cols, channels = img2.shape
end_row = start_row + rows
end_col = start_col + cols
roi = img1[start_row:end_row, start_col:end_col]

# Now create a mask of img2 and create its inverse mask also
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of img2 in ROI
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Take only region of img2 from img2 image.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

# Put img2 in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
img1[start_row:end_row, start_col:end_col] = dst

cv2.imwrite("Img/newimg.jpg",img1)

# Display the result
cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


###I think the next step to make the above image better is to use an average blur
##in the region you place the masked obj. I could use the coordinates of