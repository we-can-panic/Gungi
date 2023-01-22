import arraymancer

let
  ctx = newContext Tensor[float32]
  x = ctx.variable(randomTensor[float32](2,3,3,4))


echo x.value.shape
