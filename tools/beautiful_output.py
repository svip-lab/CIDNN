# -*- coding: utf-8 -*-
# @Time    : 2018/8/22 下午5:22
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

# IPython color
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys

    sys.excepthook = IPython.core.ultratb.ColorTB()