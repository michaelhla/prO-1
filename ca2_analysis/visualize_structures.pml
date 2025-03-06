# PyMOL script for structure visualization
load ca2_structures/original_predicted.pdb, original
color blue, original

# Sequence 0 (RMSD: 0.00Å)
load ca2_structures/3_aligned.pdb, seq_0
color red, seq_0
align seq_0, original
group seq_0, seq_0

# Sequence 1 (RMSD: 0.02Å)
load ca2_structures/9_aligned.pdb, seq_1
color red, seq_1
align seq_1, original
group seq_1, seq_1

# Sequence 2 (RMSD: 0.02Å)
load ca2_structures/6_aligned.pdb, seq_2
color red, seq_2
align seq_2, original
group seq_2, seq_2

# Sequence 3 (RMSD: 0.04Å)
load ca2_structures/2_aligned.pdb, seq_3
color red, seq_3
align seq_3, original
group seq_3, seq_3

# Sequence 4 (RMSD: 0.05Å)
load ca2_structures/8_aligned.pdb, seq_4
color red, seq_4
align seq_4, original
group seq_4, seq_4

# Sequence 5 (RMSD: 0.07Å)
load ca2_structures/4_aligned.pdb, seq_5
color red, seq_5
align seq_5, original
group seq_5, seq_5

# Sequence 6 (RMSD: 0.10Å)
load ca2_structures/7_aligned.pdb, seq_6
color red, seq_6
align seq_6, original
group seq_6, seq_6

# Sequence 7 (RMSD: 7.07Å)
load ca2_structures/5_aligned.pdb, seq_7
color red, seq_7
align seq_7, original
group seq_7, seq_7

# Sequence 8 (RMSD: 17.50Å)
load ca2_structures/1_aligned.pdb, seq_8
color red, seq_8
align seq_8, original
group seq_8, seq_8

# Sequence 9 (RMSD: 20.45Å)
load ca2_structures/0_aligned.pdb, seq_9
color red, seq_9
align seq_9, original
group seq_9, seq_9

# Show all structures as cartoon
show cartoon
hide lines
set cartoon_transparency, 0.5, original
zoom
