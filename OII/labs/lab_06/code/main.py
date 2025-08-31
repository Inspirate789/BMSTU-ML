import cv2

filename = "pos.png"
relative_indent_threshold = 1.2
relative_text_height_threshold = 1.35

img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size. 
# Kernel size increases or decreases the area of the rectangle to be detected.
# A smaller value like (10, 10) will detect each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
res = img.copy()
rects = [cv2.boundingRect(cnt) for cnt in contours]

class ContourComparator(tuple):
    def __lt__(self, other):
        return self[1] < other[1] or self[0] < other[0]

rects.sort(key=ContourComparator)

i = 0
while i < len(rects) - 1:
    (x, y, w, h) = rects[i]
    (x_next, y_next, w_next, h_next) = rects[i+1]
    if y_next <= y + h:
        x_new = min(x, x_next)
        y_new = min(y, y_next)
        w_new = max(x + w, x_next + w_next) - x_new
        h_new = max(y + h, y_next + h_next) - y_new
        rects[i] = (x_new, y_new, w_new, h_new)
        del rects[i+1]
    else:
        i += 1
        
rects = [(x, y, w, h) for (x, y, w, h) in rects if w / h > 1]
spaces = [(
    5, 
    rects[i][1] + rects[i][3], 
    img.shape[1] - 10, 
    rects[i+1][1] - rects[i][1] - rects[i][3],
) for i in range(len(rects) - 1)]

h_min = rects[0][3]
for rect in rects:
    # print(rect)
    (x, y, w, h) = rect
    if h < h_min:
        h_min = h
    # cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)

# print(f"min h = {h_min}\n")

image_correct = True
for i in range(len(spaces) - 1):
    #print(spaces[i])
    (x, y, w, h) = spaces[i]
    (x_next, y_next, w_next, h_next) = spaces[i+1]
    text_height = rects[i+1][3]
    #cv2.rectangle(res, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if text_height/h_min > relative_text_height_threshold and \
        max(h, h_next) / min(h, h_next) >= relative_indent_threshold:
        image_correct = False
        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(res, (x_next, y_next), (x_next + w_next, y_next + h_next), (0, 0, 255), 2)

if not image_correct:
    cv2.imwrite("recognized_" + filename, res)
    print(f"image has errors, see recognized_{filename}")
else:
    print("image has no errors")
    