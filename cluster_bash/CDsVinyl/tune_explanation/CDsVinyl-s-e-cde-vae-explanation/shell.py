'''
for i in range(1,43):
    f= open("CDsVinyl-e-cde-vae-part"+str(i)+".sh","w")
    f.write("#!/usr/bin/env bash\n")
    f.write("source ~/ENV/bin/activate\n")
    f.write("cd ~/Dual-Encoder\n")
    f.write("python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl_explanation_tuning/s_e_cdevae_tuning_part{0}.csv --parameters config/CDsVinyl/s-e-cde-vae-tune-keyphrase/s-e-cde-vae-part{0}.yml --tune_explanation".format(i))
'''
f= open("run_CDsVinyl.sh","w")
f.write("#!/usr/bin/env bash")
for i in range(1,43):
    f.write("sbatch --nodes=1 --time=00:40:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-e-cde-vae-part{0}.sh\n".format(i))