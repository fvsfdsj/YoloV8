from ultralytics import YOLO
import cv2
from flask import Flask, Response, request
import base64
from flask_cors import CORS

model = YOLO('./SEAttention-train65/weights/best.pt')
app = Flask(__name__)
CORS(app)

guanBi = 0


def predict():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results_list = model.predict(source=frame)
        for results in results_list:
            if results.boxes is not None:
                xyxy_boxes = results.boxes.xyxy
                conf_scores = results.boxes.conf
                cls_ids = results.boxes.cls
                for box, conf, cls_id in zip(xyxy_boxes, conf_scores, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cls_id = int(cls_id)
                    label = model.names[cls_id]
                    confidence = f"{conf:.2f}"
                    # 颜色
                    rectangle_color = (0, 255, 0)
                    label_color = (0, 0, 255)
                    # 在图像上绘制矩形框和标签
                    cv2.rectangle(frame, (x1, y1), (x2, y2), rectangle_color, 2)
                    cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                label_color, 2)
        _ret, jpeg_frame = cv2.imencode('.jpg', frame)
        frame = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if guanBi == 1:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("退出")


@app.route('/shutdown')
def shutdown():
    global guanBi
    guanBi = 1
    return ''


@app.route('/video_feed')
def video_feed():
    global guanBi
    guanBi = 0
    return Response(predict(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/image', methods=['POST'])
def predict_img():
    print(1)
    file = request.files['file']
    file.save('./temp.png')
    results_list = model.predict('./temp.png', save=True)
    # COPY FILE
    list_info = []
    for results in results_list:
        if results.boxes is not None:
            xyxy_boxes = results.boxes.xyxy
            conf_scores = results.boxes.conf
            cls_ids = results.boxes.cls

            for box, conf, cls_id in zip(xyxy_boxes, conf_scores, cls_ids):
                cls_id = int(cls_id)
                label = model.names[cls_id]
                confidence = f"{conf:.2f}"
                list_info.append({'label': label, 'value': confidence})
    with open('./temp.png', 'rb') as img_file:
        img = base64.b64encode(img_file.read()).decode('utf-8')
    return {'img': img, 'info': list_info}


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
