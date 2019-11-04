epoch_explanation = [300,400,500,600,700,800]
lambda_latent = [2.0, 1.0, 0.5, 0.3, 0.1, 0.01, 0.001]
i = 0
for ee in epoch_explanation:
    for ll in lambda_latent:
        i +=1
        f= open("./s-e-cde-vae-part"+str(i)+".yml","w")
        f.write("parameters:\n")
        f.write("\tepoch_rating: [400]\n")
        f.write("\tepoch_explanation: [{0}]\n".format(ee))
        f.write("\tbeta: [0.0001]\n")
        f.write("\tlambda_l2: [0.0001]\n")
        f.write("\tlambda_keyphrase: [1.0]\n")
        f.write("\tlabmda_latent: [{0}]\n".format(ll))
        f.write("\tlambda_rating: [1.0]\n")
        f.write("\toptimizer: [Adam]\n")
        f.write("\tlearning_rate: [0.0001]\n")
        f.write("\ttopK: [5, 10, 15, 20, 50]\n")
        f.write("\tmetric: [R-Precision, NDCG, Precision, Recall, MAP]\n")