from neural_constr import NeuralConstraintFunction
nado_file = open('./dump/MT/fisher_10k_8-4-0.60.log', 'r')
base_file = open('./dump/MT/fisher_base.log', 'r')
ref_file = open('./fluent-fisher/noids/test.noid.cleaned_0', 'r')
output_file = open('different.log', 'a')

constraint_function = NeuralConstraintFunction()
constraint_function.init_FUDGE_formality()

cnt = 1
for line1, line2, line3 in zip(nado_file, base_file, ref_file):
    v1 = constraint_function(line1)
    v2 = constraint_function(line2)
    if v1 < v2:
        output_file.write("%d\n%s\n%s\n%s\n\n\n"%(cnt, line1, line2, line3))
    cnt += 1
