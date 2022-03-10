import cv2
import os
import time
import numpy as np
class Yolo:

    def __init__(self):
        configPath = "model/yolov4-csp-swish.cfg"  # Path to cfg
        weightPath = "model/yolov4-csp-swish.weights"  # Path to weights
        metaPath = "data/coco.names"  # Path to meta data
        self.outputDir = "output/"
        frameSize = 640
        self.CAP_BUFFER_SIZE = 3
        if not os.path.exists(configPath):  # Checks whether file exists otherwise return ValueError
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath) + "`")
        with open(metaPath, 'r') as f:
            self.classes = f.read().splitlines()
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightPath)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255, size=(frameSize, frameSize), swapRB=True, crop=False)
        self.objectmap = {}

    def detectImage(self, imagePath):
        self.img = cv2.imread(imagePath)
        if self.img is None:
        # Read error handling
            raise Exception("wrong path to image")
        start = time.time()
        classIds, scores, boxes = self.model.detect(self.img, confThreshold=0.6, nmsThreshold=0.4)
        end = time.time()
        print('Current frame took {:.5f} seconds'.format(end - start))
        copy_image = self.img.copy()
        for (classId, score, box) in zip(classIds, scores, boxes):
            xmin = box[0]
            ymin = box[1]
            xmax = box[0] + box[2]
            ymax = box[1] + box[3]
            # Draw bounding boxes
            cv2.rectangle(copy_image, (xmin, ymin), (xmax, ymax),
                          color=(0, 255, 0), thickness=2)
            # copy_image = self.img.copy()
            objectClass = self.classes[classId]
            text = '%s: %.2f' % (objectClass, score)
            cv2.putText(copy_image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(0, 255, 0), thickness=2)
            self.objectmap[objectClass] = xmin, ymin, xmax, ymax
        #filename
        cv2.imwrite(self.outputDir+'outputImage.jpg', copy_image)
        cv2.imshow('Image', copy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detectVideo(self):
        #cap = cv2.VideoCapture(0)  # Uncomment to use Webcam
        cap = cv2.VideoCapture("test/test2.mp4")  # Local Stored video detection - Set input video
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.CAP_BUFFER_SIZE)
        frame_width = int(cap.get(3))  # Returns the width and height of capture video
        frame_height = int(cap.get(4))
        frame_rate = int(cap.get(5))
        # Set out for video writer
        out = cv2.VideoWriter(  # Set the Output path for video writer
            self.outputDir+"output.avi", cv2.VideoWriter_fourcc(*"XVID"), frame_rate,
            (frame_width, frame_height))
        print("Starting the YOLO loop...")

        while cap.isOpened():  # Load the input frame and write output frame.
            ret, frame_read = cap.read()  # Capture frame and return true if frame present
            # For Assertion Failed Error in OpenCV
            if not ret:  # Check if frame present otherwise he break the while loop
                print("break")
                break

            start = time.time()
            classIds, scores, boxes = self.model.detect(frame_read, confThreshold=0.6, nmsThreshold=0.4)
            end = time.time()
            print('Current frame took {:.5f} seconds'.format(end - start))
            for (classId, score, box) in zip(classIds, scores, boxes):
                cv2.rectangle(frame_read, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                              color=(0, 255, 0), thickness=2)
                # c = classes[classId[0]]
                # print(classId)
                objectClass = self.classes[classId]

                text = '%s: %.2f' % (objectClass, score)
                cv2.putText(frame_read, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color=(0, 255, 0), thickness=2)

            out.write(frame_read)
            cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
            cv2.imshow('YOLO v3 Real Time Detections', frame_read)

            # Breaking the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

            """
            End of:
            Showing processed frames in OpenCV Window
            """

        """
        End of:
        Reading frames in the loop
        """

        # Releasing camera
        cap.release()
        # Destroying all opened OpenCV windows
        cv2.destroyAllWindows()
        print("yolo done processing")

    def objectsList(self):
        print(self.objectmap.keys())
        return self.objectmap.keys()

    def cropImage(self, objectClass):
        if objectClass in self.objectmap.keys():
            # print(img.shape)  # Print image shape
            #cv2.imshow("original", self.img)
            xmin, ymin, xmax, ymax = self.objectmap[objectClass]

            # Cropping an image
            cropped_image = self.img[ymin:ymax, xmin:xmax]

            # Display cropped image
            #cv2.imshow("cropped", cropped_image)

            # Save the cropped image
            cv2.imwrite("Cropped Image.jpg", cropped_image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return cropped_image
        else:
            print("No such object is detected")
            return self.img

