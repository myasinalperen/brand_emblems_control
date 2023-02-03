import cv2
import numpy as np
import time


cam = cv2.VideoCapture(1)
cv2.namedWindow("test")
img_counter = 0

GOLD="GOLD.png"
DELL="DELL.png"
Brand_Name=DELL

#gold imageların tanımlanması
goldImg=cv2.imread(Brand_Name)
goldImg_gray =cv2.cvtColor(goldImg,cv2.COLOR_BGR2GRAY)
thresh, goldImg_wb = cv2.threshold(goldImg_gray, 100, 255, cv2.THRESH_BINARY)
imgCanny= cv2.Canny(goldImg,200,200)


def test(original,image_to_compare):
    image_to_compare=cv2.imread(image_to_compare)
    #image_to_compare= cv2.cvtColor(image_to_compare1, cv2.COLOR_BGR2GRAY)



    # 1) Check if 2 images are equals
    if original.shape == image_to_compare.shape:
        print("The images have same size and channels")
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)

        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The images are completely Equal")
        else:
            print("The images are NOT equal")

    # 2) Check for similarities between the 2 images

    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    # Define how similar they are

    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("Keypoints 1ST Image: " + str(len(kp_1)))
    print("Keypoints 2ND Image: " + str(len(kp_2)))
    print("GOOD Matches:", len(good_points))
    print("Renkli DOGRULUK Oranı: ", len(good_points) / number_keypoints * 100)
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

   # cv2.putText(result, "Logo Testi", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
    if (len(good_points) / number_keypoints * 100 )> 60:

        #result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        cv2.circle(result,(10,10), 10, (0,255,0), -1)


        cv2.imshow("Orijinal Resim", cv2.resize(result, None, fx=0.8, fy=0.8))
        cv2.imwrite("feature_matching.jpg", result)

    else:

       # result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        cv2.circle(result,(10,10), 10, (0,0,255), -1)
        cv2.imshow("Orijinal Resim", cv2.resize(result, None, fx=0.8, fy=0.8))
        cv2.imwrite("feature_matching.jpg", result)



def test_gray(original,image_to_compare):
    image_to_compare1=cv2.imread(image_to_compare)
    image_to_compare= cv2.cvtColor(image_to_compare1, cv2.COLOR_BGR2GRAY)





    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    # Define how similar they are

    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("Keypoints 1ST Image: " + str(len(kp_1)))
    print("Keypoints 2ND Image: " + str(len(kp_2)))
    print("GOOD Matches:", len(good_points))
    print("Gri DOGRULUK Oranı: ", len(good_points) / number_keypoints * 100)
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

    #cv2.putText(result, "Logo Testi", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
    if (len(good_points) / number_keypoints * 100 )> 60:

        #result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        cv2.circle(result,(10,10), 10, (0,255,0), -1)


        cv2.imshow("Gri Filtre", cv2.resize(result, None, fx=0.8, fy=0.8))
        cv2.imwrite("feature_matching.jpg", result)

    else:

       # result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        cv2.circle(result,(10,10), 10, (0,0,255), -1)
        cv2.imshow("Gri Filtre", cv2.resize(result, None, fx=0.8, fy=0.8))
        cv2.imwrite("feature_matching.jpg", result)


def test_black_white(original,image_to_compare):
    image_to_compare2=cv2.imread(image_to_compare)
    image_to_compare_gray= cv2.cvtColor(image_to_compare2, cv2.COLOR_BGR2GRAY)
    thresh, image_to_compare = cv2.threshold(image_to_compare_gray,100,255,cv2.THRESH_BINARY)



    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    # Define how similar they are

    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("Keypoints 1ST Image: " + str(len(kp_1)))
    print("Keypoints 2ND Image: " + str(len(kp_2)))
    print("GOOD Matches:", len(good_points))
    print("Siyah Beyaz DOGRULUK Oranı: ", len(good_points) / number_keypoints * 100)
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

    #cv2.putText(result, "Logo Testi", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
    if (len(good_points) / number_keypoints * 100 )> 60:

        #result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        cv2.circle(result,(10,10), 10, (0,255,0), -1)


        cv2.imshow("Siyah Beyaz Filtre", cv2.resize(result, None, fx=0.8, fy=0.8))
        cv2.imwrite("feature_matching.jpg", result)

    else:

       # result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        cv2.circle(result, (10, 10), 5, (0, 0, 255), -1)
        cv2.imshow("Siyah Beyaz Filtre", cv2.resize(result, None, fx=0.8, fy=0.8))
        cv2.imwrite("feature_matching.jpg", result)


def test_canny(original,image_to_compare):
    image_to_compare2=cv2.imread(image_to_compare)
    image_to_compare_gray= cv2.cvtColor(image_to_compare2, cv2.COLOR_BGR2GRAY)
    thresh, image_to_compare_wb = cv2.threshold(image_to_compare_gray,100,255,cv2.THRESH_BINARY)
    image_to_compare=cv2.Canny(image_to_compare_gray,200,200)


    # 2) Check for similarities between the 2 images
    print("test2")
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    # Define how similar they are

    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("Keypoints 1ST Image: " + str(len(kp_1)))
    print("Keypoints 2ND Image: " + str(len(kp_2)))
    print("GOOD Matches:", len(good_points))
    print("Canny Doğruluk Oranı: ", len(good_points) / number_keypoints * 100)
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

    #cv2.putText(result, "Logo Testi", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
    if (len(good_points) / number_keypoints * 100 )> 60:
        print("test success")
        #result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        cv2.circle(result, (10, 10), 5, (0, 255 , 0), -1)


        cv2.imshow("canny", cv2.resize(result, None, fx=0.8, fy=0.8))
        cv2.imwrite("feature_matching.jpg", result)

    else:
        print("test failed")
       # result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        cv2.circle(result, (10, 10), 5, (0, 0, 255), -1)
        cv2.imshow("Canny", cv2.resize(result, None, fx=0.8, fy=0.8))
        cv2.imwrite("feature_matching.jpg", result)

    a = len(good_points) / number_keypoints * 100
    b = round(a, 2)
    return b


def all_test(original,image_to_compare):
    start_time_gray = time.time()
    test_gray(goldImg_gray, img_name)
    end_time_gray = time.time()
    execution_time_gray = end_time_gray - start_time_gray

    start_time_wb = time.time()
    test_black_white(goldImg_wb, img_name)
    end_time_wb = time.time()
    execution_time_wb = end_time_wb - start_time_wb




while True:
    ret, frame = cam.read()
    width=int(cam.get(3))
    height=int(cam.get(3))
    if not ret:
        print("failed to grab frame")
        break

    camera = cv2.rectangle(frame, (int ((width/2)-220),int((height/2)-100) ), (int((width/2)+10),int((height/2)-20)), (0, 255, 0), 2)

    cropped_image = frame[220:300, 100:320]
    cv2.imshow("cropped", cropped_image)

    cv2.imshow("test", camera)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("ECS tuşu,kapatiliyor...")
        break
    elif k%256 == 32:
        # SPACE pressed
        start_time=time.time()
        img_name = "opencv_frame_{}.png".format(img_counter)
        #cv2.imwrite(img_name, frame)
        cv2.imwrite(img_name, cropped_image)
        print("{} written!".format(img_name))
        img_counter += 1
        test(goldImg,img_name)
        end_time=time.time()
        execution_time= end_time-start_time


        start_time_gray = time.time()
        test_gray(goldImg_gray, img_name)
        end_time_gray= time.time()
        execution_time_gray = end_time_gray- start_time_gray

        start_time_wb = time.time()
        test_black_white(goldImg_wb, img_name)
        end_time_wb = time.time()
        execution_time_wb = end_time_wb - start_time_wb

        start_time_canny = time.time()
        test_canny(imgCanny, img_name)
        end_time_canny = time.time()
        execution_time_canny = end_time_canny - start_time_canny

        print("Ex.Time Renkli         :{}".format(execution_time))
        print("Ex.Time Gri         :{}".format(execution_time_gray))
        print("Ex.Time Siyah ve Beyaz :{}".format(execution_time_wb))
        print("Ex.Time Canny :{}".format(execution_time_canny))
        print("Zaman Kazancı 1:{}".format(execution_time - execution_time_gray))
        print("Zaman Kazancı 2 :{}".format(execution_time - execution_time_wb))
        print("Zaman Kazancı 3 :{}".format(execution_time_gray - execution_time_wb))




cam.release()

cv2.destroyAllWindows()




