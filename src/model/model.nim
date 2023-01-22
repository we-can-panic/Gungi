import arraymancer

let
  (H, W, T, P) = (9, 9, 3, 38)

let
  ctx = newContext Tensor[float32]

  x = ctx.variable(randomTensor[float32])

network Agent:
  layers:
    x: Input([T, H, W, P])
    cv1: Conv2D(x.out_shape, )

    conv2d[TT](
      input, weight: Variable[TT];
      bias: Variable[TT] = nil;
      padding: Size2D = (0, 0);
      stride: Size2D = (1, 1)
    ): Variable[TT]

  forward x:
    x.cv1.relu
     .cv2.relu
     .cv3.relu
     .cv4.relu
     .cv5.relu
     .cv6.relu
     .cv7.relu
     .cv8.relu
     .cv9.relu
     .cv10.relu
     .cv11.relu
     .cv12.relu
     .cv13.relu
     .fc1

let
  agent = ctx.init(Agent)
  optim = agent.optimizer(SGDMomentum(momentum=0.9))
