import cv2
import os
import time
from queue import PriorityQueue
import numpy as np


class Yolo_Pick:

    def __init__(self):
        self.buffer = None
        configPath = "model/yolov4-csp-swish.cfg"  # Path to cfg
        weightPath = "model/yolov4-csp-swish.weights"  # Path to weights
        metaPath = "data/coco.names"  # Path to meta data
        self.outputDir = "output/"
        frameSize = 640
        self.bufferSize = 3
        self.CAP_BUFFER_SIZE = 3
        self.findFrameInterval = 2
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
        img = cv2.imread(imagePath)
        if img is None:
            # Read error handling
            raise Exception("wrong path to image")
        start = time.time()
        classIds, scores, boxes = self.model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
        end = time.time()
        print('Current frame took {:.5f} seconds'.format(end - start))
        copy_image = img.copy()
        self.drawRec(copy_image, classIds, scores, boxes)
        avgScore = np.sum(scores) / len(scores)
        self.buffer = PriorityQueue(1)
        self.buffer.put((avgScore, img, copy_image, classIds, classIds, scores, boxes))
        # filename
        cv2.imwrite(self.outputDir + 'outputImage.jpg', copy_image)
        cv2.imshow('Image', copy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detectVideo(self):
        # cap = cv2.VideoCapture(0)  # Uncomment to use Webcam
        cap = cv2.VideoCapture("test/test2.mp4")  # Local Stored video detection - Set input video
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.CAP_BUFFER_SIZE)
        frameNum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_width = int(cap.get(3))  # Returns the width and height of capture video
        frame_height = int(cap.get(4))
        frame_rate = int(cap.get(5))
        self.buffer = PriorityQueue(self.bufferSize)
        # Set out for video writer
        out = cv2.VideoWriter(  # Set the Output path for video writer
            self.outputDir + "output.avi", cv2.VideoWriter_fourcc(*"XVID"), frame_rate,
            (frame_width, frame_height))
        print("Starting the YOLO loop...")
        count = 1
        maxAvgScore = 0
        bestFrame = ()
        currentFrame = 0
        while cap.isOpened():  # Load the input frame and write output frame.
            ret, frame_read = cap.read()  # Capture frame and return true if frame present
            # For Assertion Failed Error in OpenCV
            if not ret:  # Check if frame present otherwise he break the while loop
                print("break")
                break
            frame_ori = frame_read.copy()

            start = time.time()
            classIds, scores, boxes = self.model.detect(frame_read, confThreshold=0.6, nmsThreshold=0.4)
            end = time.time()
            print('Current frame took {:.5f} seconds'.format(end - start))
            self.drawRec(frame_read, classIds, scores, boxes)

            avgScore = np.sum(scores) / len(scores)
            if avgScore > maxAvgScore:
                maxAvgScore = avgScore
                bestFrame = (avgScore, frame_ori, frame_read, classIds, scores, boxes)

            if currentFrame == frameNum or count == self.findFrameInterval:
                print('Max confidence score: ' + str(maxAvgScore))
                if not self.buffer.full():
                    # self.buffer.put((avgScore, frame_ori, frame_read, classIds, classIds, scores, boxes))
                    self.buffer.put(bestFrame)
                else:
                    lowest = self.buffer.get()
                    if lowest[0] <= avgScore:
                        # self.buffer.put((avgScore, frame_ori, frame_read,classIds, classIds, scores, boxes))
                        self.buffer.put(bestFrame)
                    else:
                        self.buffer.put(lowest)
                count = 0
                maxAvgScore = 0
            count += 1
            currentFrame += 1
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

    def drawRec(self, frame_read, classIds, scores, boxes):
        objectList = set()
        num = 1
        for (classId, score, box) in zip(classIds, scores, boxes):
            cv2.rectangle(frame_read, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                          color=(0, 255, 0), thickness=2)
            objectClass = self.classes[classId]
            name = ''
            if objectClass in objectList:
                name = objectClass + str(num)
                objectList.add(name)
                num += 1
            else:
                name = objectClass
                objectList.add(name)

            text = '%s: %.2f' % (name, score)
            cv2.putText(frame_read, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(0, 255, 0), thickness=2)

    def showBufferImg(self):
        imageNum = 1
        for frame in self.buffer.queue:
            # cv2.imwrite(self.outputDir + 'frame' + str(imageNum)+'jpg', frame[2])
            cv2.imwrite(self.outputDir + 'frame' + str(imageNum) + '.jpg', frame[2])
            cv2.imshow('Image', frame[2])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            imageNum += 1

    def objectsList(self, framenum=1):
        frame = self.buffer.queue[framenum - 1]
        self.objectmap = {}
        num = 1
        for (classId, score, box) in zip(frame[3], frame[4], frame[5]):
            c = self.classes[classId]
            xmin = box[0]
            ymin = box[1]
            xmax = box[0] + box[2]
            ymax = box[1] + box[3]
            if c in self.objectmap:
                newName = c + str(num)
                self.objectmap[newName] = (xmin, ymin, xmax, ymax)
                num += 1
            else:
                self.objectmap[c] = (xmin, ymin, xmax, ymax)
        print(self.objectmap.keys())
        return self.objectmap.keys()

    # def cropImage(self, framenum = 1, objectClass = ''):
    #     #     frame = self.buffer.queue[framenum - 1]
    #     #     for (classId, score, box) in zip(frame[3], frame[4], frame[5]):
    #     #         if self.classes[classId] == objectClass:
    #     #             xmin = box[0]
    #     #             ymin = box[1]
    #     #             xmax = box[0] + box[2]
    #     #             ymax = box[1] + box[3]
    #     #             # Cropping an image
    #     #             cropped_image = frame[1][ymin:ymax, xmin:xmax]
    #     #
    #     #             # Display cropped image
    #     #             cv2.imshow("cropped", cropped_image)
    #     #
    #     #             # Save the cropped image
    #     #             cv2.imwrite(self.outputDir + "Cropped Image.jpg", cropped_image)
    #     #
    #     #             cv2.waitKey(0)
    #     #             cv2.destroyAllWindows()
    #     #             return cropped_image
    #     #         else:
    #     #             print("No such object is detected")
    #     #             return frame[1]

    def cropImage(self, framenum=1, objectkey='person'):
        frame = self.buffer.queue[framenum - 1]
        if objectkey in self.objectmap:
            xmin = self.objectmap[objectkey][0]
            ymin = self.objectmap[objectkey][1]
            xmax = self.objectmap[objectkey][2]
            ymax = self.objectmap[objectkey][3]

            # Cropping an image
            cropped_image = frame[1][ymin:ymax, xmin:xmax]

            # Display cropped image
            cv2.imshow("cropped", cropped_image)

            # Save the cropped image
            cv2.imwrite(self.outputDir + "Cropped Image.jpg", cropped_image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return cropped_image
        else:
            print("No such object is detected")
            return frame[1]
