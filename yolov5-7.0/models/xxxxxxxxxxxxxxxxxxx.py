class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=4, ch=()):  # detection layer  //这边我把nc改成了4，原来nc是80
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW

        # if self.export and self.format == 'rknn':
        #     y = []
        #     for i in range(self.nl):
        #         y.append(self.cv2[i](x[i]))
        #         cls = torch.sigmoid(self.cv3[i](x[i]))
        #         cls_sum = torch.clamp(cls.sum(1, keepdim=True), 0, 1)
        #         y.append(cls)
        #         y.append(cls_sum)
        #     return y
                # 导出 onnx 增加
        y = []
        for i in range(self.nl):
            t1 = self.cv2[i](x[i])
            t2 = self.cv3[i](x[i])
            y.append(t1)
            y.append(t2)
        return y


        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
    


def _load(self, weights: str, task=None):
    """
    Initializes a new model and infers the task type from the model head.

    Args:
        weights (str): model checkpoint to be loaded
        task (str | None): model task
    """
    suffix = Path(weights).suffix
    if suffix == '.pt':
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.task = self.model.args['task']
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
        self.ckpt_path = self.model.pt_path
    else:
        weights = check_file(weights)
        self.model, self.ckpt = weights, None
        self.task = task or guess_model_task(weights)
        self.ckpt_path = weights
    self.overrides['model'] = weights
    self.overrides['task'] = self.task
    
    print("===========  onnx =========== ")
    import torch
    dummy_input = torch.randn(1, 3, 640, 640)
    input_names = ["data"]
    output_names = ["reg1", "cls1", "reg2", "cls2", "reg3", "cls3"]
    torch.onnx.export(self.model, dummy_input, "./weights/mybestyolov8_opset9.onnx", verbose=False, input_names=input_names, output_names=output_names, opset_version=9)
    print("======================== convert onnx Finished! .... ")

