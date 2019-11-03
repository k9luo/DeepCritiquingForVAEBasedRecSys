for i in range(6,65):
    f= open("CDsVinyl-e-cde-vae-part"+str(i)+".sh","w")
    f.write("#!/usr/bin/env bash\n")
    f.write("source ~/ENV/bin/activate\n")
    f.write("cd ~/Dual-Encoder\n")
    f.write("python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl_rating_tuning/s_e_cdevae_tuning_part{0}.csv --parameters config/CDsVinyl/s-e-cde-vae-tune-rating/s-e-cde-vae-part{0}.yml".format(i))