from models.cdevae import cde_vae
from models.e_cdevae import e_cde_vae


models = {"CDE-VAE": cde_vae,
          "E-CDE-VAE": e_cde_vae}

critiquing_models = {"E-CDE-VAE": e_cde_vae}

