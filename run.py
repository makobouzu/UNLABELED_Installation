import sys
import cv2
from PIL import Image, ImageDraw
from yolov2.utils import *
from yolov2.darknet import Darknet
from deeplab.utils import *
from pythonosc import osc_message_builder
from pythonosc import udp_client

def make_osc(input_list):
    msg = osc_message_builder.OscMessageBuilder(address= "/bbox")
    for values in input_list:
        for value in values:
            msg.add_arg(value)
    msg = msg.build()
    return msg

def detect(cfgfile, weightfile, namesfile):
    yolo = Darknet(cfgfile)

    yolo.print_network()
    yolo.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    deeplab_model = utils.load_model()

    use_cuda = 1
    if use_cuda:
        yolo.cuda()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        img = frame
        sized = cv2.resize(img, (yolo.width, yolo.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(yolo, sized, 0.5, 0.4, use_cuda)

        class_names = load_class_names(namesfile)
        bbox_info = get_boxes_info(Image.fromarray(img), boxes, class_names=class_names)
        osc_msg = make_osc(bbox_info)
        client.send_message("/index", len(bbox_info))
        client.send(osc_msg)


        mask = np.zeros_like(img)
        for box in bbox_info:
            cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), thickness=-1)
        bbox_mask = cv2.bitwise_and(img, mask)

        img = plot_boxes(Image.fromarray(img), boxes, class_names=class_names)

        labels = utils.get_pred(bbox_mask, deeplab_model)
        segment_mask = labels == 15
        segment_mask = np.repeat(segment_mask[:, :, np.newaxis], 3, axis = 2)

        output = (segment_mask * 255).astype("uint8")
        view = np.hstack((img, bbox_mask, output))

        cv2.imshow('Result', view)

        key = cv2.waitKey(100)
        if key == 27: # Press Esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    client = udp_client.SimpleUDPClient("127.0.0.1", 3296)
    detect('yolov2/cfg/yolo.cfg', 'yolov2/weights/yolo.weights', 'yolov2/data/coco.names')
