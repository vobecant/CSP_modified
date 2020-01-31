def plot_boxes(img_fname, det):

    for *xyxy, conf, _, cls in det:
        label = '%s %.2f' % ('person', conf)
        xyxy = [int(val.item()) for val in xyxy]
        plot_one_box(xyxy, img_copy, label=label, color=colors_detections[int(cls)])