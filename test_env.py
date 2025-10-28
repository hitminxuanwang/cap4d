import numpy as np, scipy, sys, inspect
import numpy.ma as ma
print("python:", sys.version.splitlines()[0])
print("numpy:", np.__version__, np.__file__)
print("scipy:", scipy.__version__, scipy.__file__)
print("ma.nomask:", ma.nomask, "type:", type(ma.nomask))
# 尝试执行导致错误的导入链的一步（仅导入，不运行你的程序）
try:
    import scipy.sparse
    print("scipy.sparse import OK")
except Exception as e:
    print("scipy.sparse import FAILED:", repr(e))
