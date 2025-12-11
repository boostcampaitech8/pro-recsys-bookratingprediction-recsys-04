# from .FM import FactorizationMachine as FM
# from .FFM import FieldAwareFactorizationMachine as FFM
# from .DeepFM import DeepFM
from .NCF import NeuralCollaborativeFilteringWithBias as NCF_B
from .MF import GeneralMatrixFactorizeModel as MF
from .MF_SVDPlus import MFSVDPlusModel as MF_SVD
from .DualHybridGMF import DualHybridGMFModel as Dual_GMF
# from .NCF import NeuralCollaborativeFiltering as NCF
# from .WDN import WideAndDeep as WDN
# from .DCN import DeepCrossNetwork as DCN
# from .FM_Image import Image_FM, Image_DeepFM, ResNet_DeepFM
# from .FM_Text import Text_FM, Text_DeepFM
from .CatBoost import CatBoost
from .XGBoost import XGBoost
from .VAE import VAE

