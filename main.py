# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
from YoloClass import Yolo
from YoloClass_save_frames import Yolo_Pick
import ImageProcessing


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = Yolo_Pick()
    #model.detectImage('test/example.jpeg')
    model.detectVideo()
    model.showBufferImg()
    model.objectsList()
    croppedImage = model.cropImage()
    #

    # model.objectsList()
    # cropped_image = model.cropImage('diningtable')
    # cv2.imshow("cropped", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
