from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='simple_gaussian_reasterization',
    ext_modules=[
        CUDAExtension('simple_gaussian_reasterization', [
            'ext.cpp',
            'render.cu',
            'render_with_jacobian.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })