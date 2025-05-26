import paddle



# paddle.argmin(x=Tensor([3, 3],"int64"), dtype="int32", )

a = paddle.arange(9).reshape([3,3]).cast('int64')
b = paddle.argmin(a, -3, dtype="int32") 
print(a) 
print(b)