epoch_explanation = [300,400,500,600,700,800]
lambda_latent = [2.0, 1.0, 0.5, 0.3, 0.1, 0.01, 0.001]
i = 0
for ee in epoch_explanation:
    for ll in lambda_latent:
        i +=1
        f= open("./s-e-cde-vae-part"+str(i)+".yml","w")
        f.write("parameters:\n")
        f.write("    epoch_rating: [400]\n")
        f.write("    epoch_explanation: [{0}]\n".format(ee))
        f.write("    beta: [0.0001]\n")
        f.write("    lambda_l2: [0.0001]\n")
        f.write("    lambda_keyphrase: [1.0]\n")
        f.write("    labmda_latent: [{0}]\n".format(ll))
        f.write("    lambda_rating: [1.0]\n")
        f.write("    optimizer: [Adam]\n")
        f.write("    learning_rate: [0.0001]\n")
        f.write("    topK: [5, 10, 15, 20, 50]\n")
        f.write("    metric: [R-Precision, NDCG, Precision, Recall, MAP]\n")