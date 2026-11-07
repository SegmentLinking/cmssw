all: auto

auto: std_sym intr_sym

# The generated headers are included by MatriplexSym.h, in interface/portable/Matriplex.
OUT := ../../interface/portable/Matriplex

clean:
	rm -f ${OUT}/std_sym_*.ah ${OUT}/intr_sym_*.ah

distclean: clean

# ---------------------------------------------------------------- #

std_sym:  ${OUT}/std_sym_3x3.ah ${OUT}/std_sym_6x6.ah

intr_sym: ${OUT}/intr_sym_3x3.ah ${OUT}/intr_sym_6x6.ah


# ================================================================ #

GM := ./gen_mul.pl

${OUT}/std_sym_3x3.ah:
	${GM} "mult_sym(3);" > $@

${OUT}/std_sym_6x6.ah:
	${GM} "mult_sym(6);" > $@

${OUT}/intr_sym_3x3.ah:
	${GM} "mult_sym_fma_intrinsic(3);" > $@

${OUT}/intr_sym_6x6.ah:
	${GM} "mult_sym_fma_intrinsic(6);" > $@
