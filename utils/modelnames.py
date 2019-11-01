from models.cdevae import cde_vae
from models.e_cdevae import e_cde_vae
from models.s_e_cdevae import s_e_cde_vae


models = {"CDE-VAE": cde_vae,
          "E-CDE-VAE": e_cde_vae,
          "S-E-CDE-VAE": s_e_cde_vae}

critiquing_models = {"E-CDE-VAE": e_cde_vae,
                     "S-E-CDE-VAE": s_e_cde_vae}

