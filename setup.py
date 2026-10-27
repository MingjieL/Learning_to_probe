from pathlib import Path
import platform

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(".")


class PrefixBuildExtension(BuildExtension):
    """Compile ISA-specific row kernels without changing the baseline module ISA."""

    def __init__(self, *args, **kwargs):
        kwargs["use_ninja"] = False
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        original_compile = self.compiler._compile

        def compile_source(obj, src, ext, cc_args, post_args, pp_opts):
            flags = list(post_args)
            name = Path(src).stem
            if platform.machine().lower() in ("x86_64", "amd64") and name in (
                "_prefix_softmax_avx2", "_prefix_softmax_avx512",
            ):
                capability = "AVX512" if name.endswith("avx512") else "AVX2"
                flags += ["-mavx2", "-mfma", "-fno-fast-math", "-ffp-contract=off",
                          f"-DCPU_CAPABILITY={capability}", f"-DCPU_CAPABILITY_{capability}"]
                if capability == "AVX512":
                    flags += ["-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl"]
            original_compile(obj, src, ext, cc_args, flags, pp_opts)

        self.compiler._compile = compile_source
        try:
            super().build_extensions()
        finally:
            self.compiler._compile = original_compile

setup(
    ext_modules=[
        CppExtension(
            "ft_anns.adaptive_query._gain_cost_native",
            [str(ROOT / "src/ft_anns/adaptive_query" / source) for source in (
                "_gain_cost_native.cpp", "_prefix_softmax_avx2.cpp", "_prefix_softmax_avx512.cpp")],
            depends=[str(ROOT / "src/ft_anns/adaptive_query" / header) for header in (
                "_prefix_softmax.h", "_prefix_softmax_vector.h", "_positive_marginal.h", "_positive_scan.h")],
            extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": PrefixBuildExtension},
)
